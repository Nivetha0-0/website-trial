# forwarding.py
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from typing import TYPE_CHECKING, Optional

# Import Client only for type checking to avoid potential runtime circular imports
# This is the actual class that firestore.client() returns an instance of.
if TYPE_CHECKING:
    from google.cloud.firestore import Client as FirestoreClient


def save_unanswered_question(db: "FirestoreClient", question_english: str):
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
            
            
    except Exception as e:
        pass


def save_user_interaction(db: "FirestoreClient", question_english: str, answer_english: str, user_session_id: str = "anonymous"):
    try:
        # Create document data
        interaction_data = {
            "question": question_english,
            "answer": answer_english,
            "timestamp": datetime.now(),
            "session_id": user_session_id,
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
        
        db.collection("user").add(interaction_data)
        
    except Exception as e:
        pass