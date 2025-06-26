# main_trial.py
import streamlit as st
from streamlit_chat import message
import os
import numpy as np
import hashlib
import json
import tempfile

# Langchain and Pydantic imports
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from pymongo.mongo_client import MongoClient
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, SecretStr
from typing import Literal, Optional, List, cast

# Google Cloud imports
# IMPORTANT CHANGE HERE: Use translate_v3beta1 consistently
from google.cloud import translate_v3beta1 as translate # <--- MODIFIED THIS LINE
from google.cloud import texttospeech
from google.cloud import speech

# Import your translation and speech utilities (NOTE: transcribe_audio is removed from this line)
from trans_trials import get_translator_client, get_texttospeech_client, get_speech_client, translate_text, get_supported_languages

# Import the new speech input component
from speech_input_component import audio_input_widget


# --- Google Cloud API Setup ---
GOOGLE_CLOUD_KEY_PATH: Optional[str] = None # Initialize to None

try:
    gcp_sa_key_json_content = st.secrets["gcp_service_account_key"]
except KeyError:
    st.error("Secret 'gcp_service_account_key' not found. Please add the content of your Google Service Account JSON key to Streamlit secrets.")
    st.stop()

if gcp_sa_key_json_content:
    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp_file:
            temp_file.write(gcp_sa_key_json_content)
            temp_file.flush()
            GOOGLE_CLOUD_KEY_PATH = temp_file.name
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CLOUD_KEY_PATH
    except Exception as e:
        st.error(f"Error handling Google Cloud credentials: {e}")
        st.stop()
else:
    st.error("Google Cloud service account key content is missing from secrets. Cannot initialize GCP.")
    st.stop()


_GC_PROJECT_ID: str
try:
    _GC_PROJECT_ID = st.secrets["GOOGLE_CLOUD_PROJECT"]
except KeyError:
    st.error("Secret 'GOOGLE_CLOUD_PROJECT' not found. This is required for Translation API V3. Please add it to your Streamlit secrets.")
    st.stop()


translator_client: Optional[translate.TranslationServiceClient] = get_translator_client(GOOGLE_CLOUD_KEY_PATH)
if not translator_client:
    st.warning("Google Cloud Translator client could not be initialized. Translation features will be limited.")

texttospeech_client: Optional[texttospeech.TextToSpeechClient] = get_texttospeech_client(GOOGLE_CLOUD_KEY_PATH)
if not texttospeech_client:
    st.warning("Google Cloud Text-to-Speech client could not be initialized. Audio output will be disabled.")

speech_client: Optional[speech.SpeechClient] = get_speech_client(GOOGLE_CLOUD_KEY_PATH)
if not speech_client:
    st.warning("Google Cloud Speech-to-Text client could not be initialized. Voice input features will be disabled.")


# Define the list of languages you want to support
ALLOWED_LANGUAGES: List[str] = ['en', 'ta', 'te', 'hi'] # English, Tamil, Telugu, Hindi

DEFAULT_LANGUAGE: Literal['en'] = "en"

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


# --- Common Logic for Processing User Input (text or speech) ---
def _process_common_logic(user_input_original: str):
    """Contains the core logic for processing a user's query after it's in text form."""
    if not user_input_original.strip():
        return # Do nothing if input is empty

    current_selected_language: str = str(st.session_state.selected_language)

    # Translate user input to English for LLM context
    # user_input_original is already in the selected_language (either typed or transcribed)
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

    bot_response: str = translate_text(translator_client, bot_response_english, current_selected_language, DEFAULT_LANGUAGE, _GC_PROJECT_ID) # type: ignore

    st.session_state.chat_history.append((user_input_original, bot_response))


# --- Input Processing Function for Text ---
def process_text_input():
    """Handles text input from the user."""
    user_text = st.session_state.user_text_input.strip()
    _process_common_logic(user_text)
    st.session_state.user_text_input = "" # Clear text input box after processing


# --- Display and Main App Structure ---
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
    # Reset any audio-related temporary state when language changes
    if 'last_audio_hash_for_stt' in st.session_state: # Clear state from speech_input_component
        st.session_state.last_audio_hash_for_stt = None
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

    chat_container = st.container()
    with chat_container:
        display_chat()

    translated_placeholder_raw = translate_text(translator_client, "Type your message here...", st.session_state.selected_language, DEFAULT_LANGUAGE, _GC_PROJECT_ID) # type: ignore
    translated_placeholder: str = translated_placeholder_raw if translated_placeholder_raw else "Type your message here..."

    # Text input field
    st.text_input(
        "Type your message",
        key="user_text_input",
        placeholder=translated_placeholder,
        on_change=process_text_input # Call specific text input handler
    )

    st.markdown("---") # Separator for clarity

    # Call the new speech input component
    # It will display the recorder and return transcribed text if available
    transcribed_text_from_speech = audio_input_widget(speech_client, st.session_state.selected_language)

    # If the audio_input_widget returned transcribed text, process it
    if transcribed_text_from_speech:
        _process_common_logic(transcribed_text_from_speech)
        # Force a rerun to update the chat history with the new question and answer
        st.rerun()


if __name__ == "__main__":
    main()