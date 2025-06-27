# forwarding.py
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

firebase_config = "key.json"

# Initialize Firebase if not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_config)
    firebase_admin.initialize_app(cred)

db = firestore.client()

def save_unanswered_question(question_english):
    """
    Save unanswered question to Firebase for doctor review
    
    Args:
        question_english (str): The user's question in English
    """
    try:
        # Reference to the doctor document
        doctor_doc_ref = db.collection("DOCTOR").document("1")
        
        # Get current data
        doc = doctor_doc_ref.get()
        data = doc.to_dict() if doc.exists else {}
        
        # Get existing questions list
        qn_list = data.get("qn", [])
        
        # Add new question if not already present
        if question_english not in qn_list:
            qn_list.append(question_english)
            
            # Update the document
            doctor_doc_ref.set({
                "qn": qn_list
            }, merge=True)
            
            print(f"✅ Unanswered question saved for doctor review: {question_english}")
        else:
            print(f"ℹ️  Question already exists in unanswered list: {question_english}")
            
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
    try:
        # Create document data
        interaction_data = {
            "question": question_english,
            "answer": answer_english,
            "timestamp": datetime.now(),
            "session_id": user_session_id or "anonymous",
            "status": "answered"
        }
        
        # Check if this should be marked as unanswered
        unanswered_indicators = [
            "doctor has been notified",
            "doctor will be notified", 
            "check back in a few days",
            "unable to answer your question"
        ]
        
        if any(indicator in answer_english.lower() for indicator in unanswered_indicators):
            interaction_data["status"] = "forwarded_to_doctor"
        
        # Add to user collection
        db.collection("user").add(interaction_data)
        
        print(f"✅ User interaction saved: Q: {question_english[:50]}...")
        
    except Exception as e:
        print(f"❌ Error saving user interaction: {str(e)}")
        raise e