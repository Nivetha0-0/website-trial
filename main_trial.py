import streamlit as st
from streamlit_chat import message
import os
import numpy as np
import hashlib
import json
import tempfile
import asyncio # New import for managing async tasks

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from pymongo.mongo_client import MongoClient
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, SecretStr
from typing import Literal, Optional, List, cast
from google.cloud import translate
from google.cloud import texttospeech
from google.cloud import speech

# Import your deployment utility functions (assuming deploy_translation.py is in the same directory)
from trans_trials import get_translator_client, get_texttospeech_client, get_speech_client, translate_text, get_supported_languages

# Import the new STT module
from speech_to_text import start_webrtc_stream, stop_webrtc_stream, IS_LISTENING as stt_module_is_listening # Import specific functions and flag


# --- Google Cloud API Setup ---
GOOGLE_CLOUD_KEY_PATH: Optional[str] = None # Initialize to None

# Retrieve GCP service account key JSON content from Streamlit secrets
gcp_sa_key_json_content: Optional[str] = None
try:
    gcp_sa_key_json_content = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]
except KeyError:
    st.error("Secret 'GOOGLE_APPLICATION_CREDENTIALS' not found. Please add the content of your Google Service Account JSON key to Streamlit secrets.")
    st.stop()

# Write the JSON content to a temporary file and set GOOGLE_APPLICATION_CREDENTIALS
if gcp_sa_key_json_content:
    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp_file:
            temp_file.write(gcp_sa_key_json_content)
            temp_file.flush() # Ensure the content is written to disk
            GOOGLE_CLOUD_KEY_PATH = temp_file.name
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CLOUD_KEY_PATH
    except Exception as e:
        st.error(f"Error handling Google Cloud credentials: {e}")
        st.stop()
else:
    st.error("Google Cloud service account key content is missing from secrets. Cannot initialize GCP.")
    st.stop()


# Get GOOGLE_CLOUD_PROJECT from Streamlit secrets
_GC_PROJECT_ID: str
try:
    _GC_PROJECT_ID = st.secrets["GOOGLE_CLOUD_PROJECT"]
except KeyError:
    st.error("Secret 'GOOGLE_CLOUD_PROJECT' not found. This is required for Translation API V3. Please add it to your Streamlit secrets.")
    st.stop()


# Initialize Google Cloud clients using the utility functions
translator_client: Optional[translate.TranslationServiceClient] = get_translator_client(GOOGLE_CLOUD_KEY_PATH)
if not translator_client:
    st.warning("Google Cloud Translator client could not be initialized. Translation features will be limited.")

texttospeech_client: Optional[texttospeech.TextToSpeechClient] = get_texttospeech_client(GOOGLE_CLOUD_KEY_PATH)
if not texttospeech_client:
    st.warning("Google Cloud Text-to-Speech client could not be initialized. Audio output will be disabled.")

speech_client: Optional[speech.SpeechClient] = get_speech_client(GOOGLE_CLOUD_KEY_PATH)
if not speech_client:
    st.warning("Google Cloud Speech-to-Text client could not be initialized. Voice input features (if implemented) will be disabled.")


# Define the list of languages you want to support
ALLOWED_LANGUAGES: List[str] = ['en', 'ta', 'te', 'hi'] # English, Tamil, Telugu, Hindi

DEFAULT_LANGUAGE: Literal['en'] = "en"

# Ensure SUPPORTED_LANGUAGES is always initialized to a dict (global scope)
SUPPORTED_LANGUAGES: dict[str, str] = {}
if not SUPPORTED_LANGUAGES:
    SUPPORTED_LANGUAGES = get_supported_languages(translator_client, _GC_PROJECT_ID, allowed_langs=ALLOWED_LANGUAGES)


# --- Cosine Similarity (Manual Implementation) ---
def cosine_similarity_manual(vec1, vec2):
    """Calculates cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

# --- LangChain Prompts and Pydantic Models ---
tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

class CasualSubject(BaseModel):
    description: str = Field(
        description="""Classify the given user query into one of two categories:
Casual Greeting - If the query is a generic greeting or social pleasantry (e.g., 'Hi', 'How are you?', 'Good morning').
Subject-Specific - If the query is about a particular topic or seeks information (e.g., 'What is Python?', 'Tell me about space travel').
Return only the category name: 'Casual Greeting' or 'Subject-Specific'.""",
    )
    category: Literal['Casual Greeting', 'Subject-Specific'] = Field(
        description="The classified category of the user query."
    )

class RelatedNot(BaseModel):
    description: str = Field(
        description="""Determine whether the given user query is related to animal bites.
Categories:
Animal Bite-Related - If the query mentions animal bites, their effects, treatment, prevention, or specific cases (e.g., 'What to do after a dog bite?', 'Are cat bites dangerous?').
Not Animal Bite-Related - If the query does not pertain to animal bites.
Return only the category name: 'Animal Bite-Related' or 'Not Animal Bite-Related'.""",
    )
    category: Literal['Animal Bite-Related', 'Not Animal Bite-Related'] = Field(
        description="The classified category regarding animal bite relevance."
    )

# --- Chatbot Initialization ---
openai_api_key_secret: Optional[SecretStr]
try:
    openai_api_key_secret = SecretStr(st.secrets["OPENAI_KEY"])
except KeyError:
    st.error("Secret 'OPENAI_KEY' not found. Please add it to your Streamlit secrets.")
    st.stop()

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key_secret)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=openai_api_key_secret)
smaller_llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=openai_api_key_secret)
larger_llm = ChatOpenAI(temperature=0, model="gpt-4o", api_key=openai_api_key_secret)

# --- MongoDB Initialization ---
mongodb_uri: Optional[str]
try:
    mongodb_uri = st.secrets["MONGODB_URI"]
except KeyError:
    st.error("Secret 'MONGODB_URI' not found. Please add it to your Streamlit secrets.")
    st.stop()

try:
    client = MongoClient(mongodb_uri)
    db = client["pdf_file"]
    collection = db["animal_bites"]
    _ = db.list_collection_names() # Test connection
except Exception as e:
    st.error(f"Failed to connect to MongoDB: {e}. Please check your MONGODB_URI and ensure MongoDB is accessible.")
    st.stop()

# --- Streamlit Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_language" not in st.session_state:
    st.session_state.selected_language = DEFAULT_LANGUAGE
if "webrtc_ctx" not in st.session_state:
    st.session_state.webrtc_ctx = None # To store the webrtc_streamer context
if "stt_task_running" not in st.session_state:
    st.session_state.stt_task_running = False # Flag to indicate if STT task is active
if "interim_stt_placeholder" not in st.session_state:
    st.session_state.interim_stt_placeholder = st.empty() # Placeholder for interim results


# --- Callback for final transcript from STT module ---
def handle_final_transcript(transcript: str):
    """
    This function is called by the stt_module when a final transcript is ready.
    """
    print(f"Main App: Received final transcript: {transcript}")
    # Store the transcript to be processed by the main app on next rerun
    st.session_state.voice_input_text = transcript
    st.session_state.stt_task_running = False # Signal that the STT session has completed for this utterance
    # Clear the text input box as the voice input is now the source
    st.session_state.user_input = ""
    st.rerun() # Trigger a rerun to process this voice input


# --- Main Processing Logic ---
def process_input():
    # Determine the source of input (text_input or voice_input_text from STT)
    user_input_original = ""
    if "voice_input_text" in st.session_state and st.session_state.voice_input_text:
        user_input_original = st.session_state.voice_input_text
        del st.session_state.voice_input_text # Clear it after use
    elif "user_input" in st.session_state:
        user_input_original = st.session_state.user_input.strip()

    if not user_input_original:
        return

    current_selected_language: str = str(st.session_state.selected_language) if st.session_state.selected_language is not None else DEFAULT_LANGUAGE

    # Translate user input to English for LLM processing
    user_input_english_raw = translate_text(translator_client, user_input_original, DEFAULT_LANGUAGE, current_selected_language, _GC_PROJECT_ID) # type: ignore
    user_input_english: str = user_input_english_raw if user_input_english_raw else user_input_original
    if not user_input_english.strip():
        user_input_english = user_input_original

    # Rephrase user input for LLM context
    retrieval_prompt_template = f"""Given a chat_history and the latest_user_input question/statement \
which MIGHT reference context in the chat history, formulate a standalone question/statement \
which can be understood without the chat history. Do NOT answer the question, \
If the latest_user_input is a pleasantry (e.g., 'thank you', 'thanks', 'got it', 'okay'), return it as is without modification. Otherwise, ensure the reformulated version is self-contained.\
chat_history: {st.session_state.chat_history}
latest_user_input:{user_input_english}"""

    modified_user_input_result = larger_llm.invoke(retrieval_prompt_template).content
    modified_user_input: str = modified_user_input_result if isinstance(modified_user_input_result, str) else ""
    if not modified_user_input.strip():
        modified_user_input = user_input_english

    # Classify query type (Casual Greeting vs. Subject-Specific)
    classification_category = 'Subject-Specific' # Default to subject-specific
    try:
        response_casual_subject = smaller_llm.with_structured_output(CasualSubject).invoke(tagging_prompt.invoke({"input": modified_user_input}))
        classification_category = response_casual_subject.category if isinstance(response_casual_subject, CasualSubject) else response_casual_subject.get('category', 'Subject-Specific')
    except Exception as e:
        st.error(f"Error classifying query type: {e}. Assuming Subject-Specific.")

    bot_response_english: Optional[str] = None
    bot_response: str = ""

    if classification_category == 'Subject-Specific':
        try:
            embedding = embeddings_model.embed_query(modified_user_input)

            result = collection.aggregate([
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embeddings",
                        "queryVector": embedding,
                        "numCandidates": 100,
                        "limit": 3
                    }
                }
            ])

            context = ""
            for doc in result:
                db_embedding = doc["embeddings"]
                val = cosine_similarity_manual(db_embedding, embedding)
                if round(val, 2) >= 0.55:
                    context = context + doc["raw_data"] + "\n\n"

            if context.strip():
                prompt_template = f"""you are a chatbot meant to answer questions related to animal bites, answer the question based on the given context.
                context:{context}
                question:{modified_user_input}"""
                response_llm_english_result = llm.invoke(prompt_template).content
                bot_response_english = response_llm_english_result if isinstance(response_llm_english_result, str) else None
            else:
                relevance_category = 'Animal Bite-Related' # Default to Animal Bite-Related
                try:
                    response_related_not = smaller_llm.with_structured_output(RelatedNot).invoke(tagging_prompt.invoke({"input": modified_user_input}))
                    relevance_category = response_related_not.category if isinstance(response_related_not, RelatedNot) else response_related_not.get('category', 'Animal Bite-Related')
                except Exception as e:
                    st.error(f"Error classifying relevance: {e}. Assuming Animal Bite-Related.")

                if relevance_category == 'Not Animal Bite-Related':
                    bot_response_english = "Sorry, but I specialize in answering questions related to animal bites.\
                                            I may not be able to help with your query, but if you have any questions about animal bites, \
                                            their effects, treatment, or prevention, I'd be happy to assist!"
                else: # relevance_category == 'Animal Bite-Related' AND no context found
                    # Directly provide the response without sending an email
                    bot_response_english = "I am unable to answer your question at the moment. The Doctor has been notified, please check back in a few days."
        except Exception as e:
            st.error(f"Error during subject-specific processing: {e}")
            bot_response_english = "An internal error occurred while processing your request. Please try again."

    else: # classification_category == 'Casual Greeting'
        try:
            bot_response_english_result = llm.invoke(f"""system:you are a friendly chatbot that specializes in medical questions related to animal bites.
                                    question: {user_input_english}""").content
            bot_response_english = bot_response_english_result if isinstance(bot_response_english_result, str) else None
        except Exception as e:
            st.error(f"Error during casual greeting processing: {e}")
            bot_response_english = "An internal error occurred while generating a greeting. Please try again."

    bot_response = translate_text(translator_client, bot_response_english, current_selected_language, DEFAULT_LANGUAGE, _GC_PROJECT_ID) # type: ignore

    st.session_state.chat_history.append((user_input_original, bot_response))
    st.session_state.user_input = "" # Clear text input after processing
    # If this was a voice input, ensure the text input is cleared as well
    if "voice_input_text" in st.session_state:
        st.session_state.user_input = "" # Clear the text input box
        del st.session_state.voice_input_text # Remove the temporary voice input


def display_chat():
    os.makedirs("tts_audio", exist_ok=True)

    for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
        message(user_msg, is_user=True, key=f"user_msg_{i}")
        message(bot_msg, key=f"bot_msg_{i}")

        text_hash = hashlib.md5(bot_msg.encode('utf-8')).hexdigest()
        audio_file_path = f"tts_audio/{text_hash}_{st.session_state.selected_language}.mp3"

        if bot_msg and texttospeech_client:
            current_lang_for_tts = str(st.session_state.selected_language)
            try:
                synthesis_input = texttospeech.SynthesisInput(text=bot_msg)
                voice_name_map = {
                    'en': 'en-US-Wavenet-C',
                    'hi': 'hi-IN-Wavenet-C',
                    'ta': 'ta-IN-Wavenet-C',
                    'te': 'te-IN-Standard-A'
                }
                voice_name = voice_name_map.get(current_lang_for_tts)
                voice = texttospeech.VoiceSelectionParams(
                    language_code=current_lang_for_tts,
                    name=voice_name if voice_name else None,
                    ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3
                )
                response_tts = texttospeech_client.synthesize_speech(
                    request={"input": synthesis_input, "voice": voice, "audio_config": audio_config}
                )
                with open(audio_file_path, "wb") as out:
                    out.write(response_tts.audio_content)
            except Exception as e:
                st.warning(f"Could not generate TTS audio: {e}. Audio playback unavailable for this message.")
                audio_file_path = None
        else:
            audio_file_path = None

        if audio_file_path and os.path.exists(audio_file_path):
            with st.container():
                audio_file = open(audio_file_path, "rb")
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3", start_time=0)


def set_language():
    st.session_state.selected_language = st.session_state.lang_selector
    st.session_state.chat_history = []
    st.rerun()


def main():
    st.set_page_config(page_title="Multilingual Animal Bites Chatbot", layout="centered")
    st.title("Chatbot for Animal Bites")

    if not SUPPORTED_LANGUAGES:
        st.error("FATAL ERROR: NO SUPPORTED LANGUAGES FOUND! Translation features will be SEVERELY limited. Please check Google Cloud credentials and API status.")
        if 'en' not in SUPPORTED_LANGUAGES:
            SUPPORTED_LANGUAGES['en'] = "English"

    lang_codes: List[str] = list(SUPPORTED_LANGUAGES.keys())

    if st.session_state.selected_language not in lang_codes:
        st.session_state.selected_language = DEFAULT_LANGUAGE

    try:
        current_lang_index = lang_codes.index(st.session_state.selected_language)
    except ValueError:
        current_lang_index = lang_codes.index(DEFAULT_LANGUAGE)

    st.sidebar.selectbox(
        "Select Language",
        options=lang_codes,
        format_func=lambda code: str(SUPPORTED_LANGUAGES.get(code, code)),
        key="lang_selector",
        on_change=set_language,
        index=current_lang_index
    )

    # --- Voice Input Section ---
    st.subheader("Voice Input")
    st.write("Click 'Start Recording' to begin speaking. Click 'Stop Recording' to end and process.")

    col1, col2 = st.columns(2)

    with col1:
        start_button = st.button("🎤 Start Recording", disabled=stt_module_is_listening or speech_client is None)
    with col2:
        stop_button = st.button("⏹ Stop Recording", disabled=not stt_module_is_listening or speech_client is None)

    # Display interim results from the STT module
    # This placeholder is passed to the STT module to update directly
    st.session_state.interim_stt_placeholder.empty()
    st.session_state.interim_stt_placeholder = st.empty() # Re-create on each rerun for correct placement


    if start_button:
        if not stt_module_is_listening and speech_client:
            st.session_state.webrtc_ctx = start_webrtc_stream(
                speech_client=speech_client,
                selected_language=st.session_state.selected_language,
                interim_display_placeholder=st.session_state.interim_stt_placeholder,
                final_transcript_callback=handle_final_transcript
            )
            st.session_state.stt_task_running = True
            st.info("Voice input active. Speak now...")
        elif speech_client is None:
            st.error("Speech-to-Text client not initialized. Please check Google Cloud credentials.")
        else:
            st.warning("Voice input is already active.")


    if stop_button:
        if stt_module_is_listening:
            stop_webrtc_stream(st.session_state.webrtc_ctx)
            st.session_state.webrtc_ctx = None
            st.session_state.stt_task_running = False
            st.info("Voice input stopped. Processing transcript...")
            # If the STT task is still running, handle_final_transcript will trigger rerun
            # If it already stopped (e.g., short utterance), we might need a manual rerun here
            # to ensure the processing of st.session_state.voice_input_text.
            # However, handle_final_transcript already does st.rerun(), so this is usually covered.
        else:
            st.warning("Voice input is not currently active.")


    # --- Chat Display and Text Input Section ---
    chat_container = st.container()
    with chat_container:
        display_chat()

    translated_placeholder_raw = translate_text(translator_client, "Type your message here", st.session_state.selected_language, DEFAULT_LANGUAGE, _GC_PROJECT_ID) # type: ignore
    translated_placeholder: str = translated_placeholder_raw if translated_placeholder_raw else "Type your message here"

    # Use a key here to ensure the text input is cleared correctly
    st.text_input(
        "Type something...",
        key="user_input",
        placeholder=translated_placeholder,
        on_change=process_input,
        # Disable text input while voice input is active to avoid conflicts
        disabled=stt_module_is_listening
    )


    # This ensures that if a final transcript was received (setting voice_input_text),
    # process_input is called to handle it.
    if "voice_input_text" in st.session_state and st.session_state.voice_input_text:
        process_input()


if __name__ == "main":
    main()
