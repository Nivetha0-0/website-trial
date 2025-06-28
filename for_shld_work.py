import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import os # Import os for checking file existence

# Global variable to track if Firebase has been initialized
_firebase_app_initialized = False

def initialize_firebase_app(firebase_config_path: str):
    """
    Initializes the Firebase Admin SDK if it hasn't been initialized already.

    Args:
        firebase_config_path (str): The path to the Firebase service account key JSON file.
    """
    global _firebase_app_initialized
    if not _firebase_app_initialized:
        try:
            if not os.path.exists(firebase_config_path):
                print(f"❌ Error: Firebase service account key file not found at {firebase_config_path}")
                raise FileNotFoundError(f"Firebase service account key file not found at {firebase_config_path}")

            cred = credentials.Certificate(firebase_config_path)
            firebase_admin.initialize_app(cred)
            _firebase_app_initialized = True
            print("✅ Firebase Admin SDK initialized successfully.")
        except Exception as e:
            print(f"❌ Error initializing Firebase Admin SDK: {e}")
            raise e
    else:
        print("ℹ️ Firebase Admin SDK already initialized.")

# Firestore client, initialized only after the app is initialized
# This will be called globally, but the actual client creation
# depends on _firebase_app_initialized being True.
# A more robust solution might involve passing the db object or
# ensuring initialization order more strictly in a larger application.
# For this context, it's assumed initialize_firebase_app is called before
# any functions that use 'db'.
try:
    # Attempt to get the Firestore client. This will only succeed if
    # initialize_firebase_app has been called successfully.
    db = firestore.client()
except ValueError:
    # This error occurs if firebase_admin.initialize_app() hasn't been called yet.
    # We will handle this by ensuring initialize_firebase_app is called first in main.py.
    print("⚠️ Firestore client not initialized yet. Ensure initialize_firebase_app is called first.")
    db = None # Set to None, functions will need to check for its existence.


def save_unanswered_question(question_english):
    """
    Save unanswered question to Firebase for doctor review

    Args:
        question_english (str): The user's question in English
    """
    if db is None:
        print("❌ Firebase database not available. Cannot save unanswered question.")
        return

    try:
        doctor_doc_ref = db.collection("DOCTOR").document("1")
        doc = doctor_doc_ref.get()
        data = doc.to_dict() if doc.exists else {}
        qn_list = data.get("qn", [])

        if question_english not in qn_list:
            qn_list.append(question_english)

            doctor_doc_ref.set({
                "qn": qn_list
            }, merge=True)

            print(f"✅ Unanswered question saved for doctor review: {question_english}")
        else:
            print(f"ℹ️ Question already exists in unanswered list: {question_english}")

    except Exception as e:
        print(f"❌ Error saving unanswered question: {str(e)}")
        raise e

def save_user_interaction(question_english, answer_english, user_session_id=None):
    """
    Save user question and bot response to Firebase

    Args:
        question_english (str): User's question in English
        answer_english (str): Bot's answer in English
        user_session_id (str, optional): User session identifier
    """
    if db is None:
        print("❌ Firebase database not available. Cannot save user interaction.")
        return

    try:
        interaction_data = {
            "question": question_english,
            "answer": answer_english,
            "timestamp": datetime.now(),
            "session_id": user_session_id or "anonymous",
            "status": "answered"
        }

        unanswered_indicators = [
            "doctor has been notified",
            "doctor will be notified",
            "check back in a few days",
            "unable to answer your question"
        ]

        if any(indicator in answer_english.lower() for indicator in unanswered_indicators):
            interaction_data["status"] = "forwarded_to_doctor"

        db.collection("user").add(interaction_data)

        print(f"✅ User interaction saved: Q: {question_english[:50]}...")

    except Exception as e:
        print(f"❌ Error saving user interaction: {str(e)}")
        raise e