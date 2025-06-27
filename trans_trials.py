import os
from google.cloud import translate 
from google.cloud import texttospeech 
from google.cloud import speech 
from typing import Optional, List, Type, TypeVar

GCClient = TypeVar('GCClient')
_translator_client = None
_texttospeech_client = None
_speech_client = None

def _initialize_gc_client(client_class: Type[GCClient], key_path: str) -> Optional[GCClient]:
    """Generic function to initialize a Google Cloud client."""
    try:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
        client = client_class()
        return client
    except Exception as e:
        print(f"Error initializing Google Cloud {client_class.__name__}: {e}")
        return None

def get_translator_client(key_path: str) -> Optional[translate.TranslationServiceClient]:
    """Returns a singleton instance of TranslationServiceClient."""
    global _translator_client
    if _translator_client is None:
        _translator_client = _initialize_gc_client(translate.TranslationServiceClient, key_path)
    return _translator_client

def get_texttospeech_client(key_path: str) -> Optional[texttospeech.TextToSpeechClient]:
    """Returns a singleton instance of TextToSpeechClient."""
    global _texttospeech_client
    if _texttospeech_client is None:
        _texttospeech_client = _initialize_gc_client(texttospeech.TextToSpeechClient, key_path)
    return _texttospeech_client

def get_speech_client(key_path: str) -> Optional[speech.SpeechClient]:
    """Returns a singleton instance of SpeechClient."""
    global _speech_client
    if _speech_client is None:
        _speech_client = _initialize_gc_client(speech.SpeechClient, key_path)
    return _speech_client

def get_supported_languages(client, project_id: str, allowed_langs: Optional[List[str]] = None) -> dict[str, str]:
    if not client:
        return {}
    try:
        parent = f"projects/{project_id}/locations/global"
        response = client.get_supported_languages(parent=parent, display_language_code='en')

        languages = {}
        for lang in response.languages:
            if allowed_langs and lang.language_code not in allowed_langs:
                continue

            display_name = lang.display_name if lang.display_name else lang.language_code
            languages[lang.language_code] = display_name
        return languages
    except Exception as e:
        print(f"Error fetching supported languages (V3): {e}")
        return {}

def translate_text(client, text: Optional[str], target_language_code: str, source_language_code: str, project_id: str) -> Optional[str]:
    if not text or not client:
        return text

    if source_language_code == target_language_code:
        return text

    try:
        parent = f"projects/{project_id}/locations/global"
        response = client.translate_text(
            request={
                "parent": parent,
                "contents": [text],
                "mime_type": "text/plain",
                "source_language_code": source_language_code,
                "target_language_code": target_language_code,
            }
        )
        translated_text = response.translations[0].translated_text
        return translated_text
    except Exception as e:
        print(f"Error translating text (V3): {e}")
        return None