# trans_trials.py

import os
from typing import Optional, Dict, List
from google.cloud import translate_v3beta1 as translate
from google.cloud import texttospeech
from google.cloud import speech


def get_translator_client(credentials_path: Optional[str]) -> Optional[translate.TranslationServiceClient]:
    """Initializes and returns a Google Cloud Translation client."""
    try:
        if credentials_path and os.path.exists(credentials_path):
            return translate.TranslationServiceClient()
        else:
            print("Warning: Google Cloud credentials path not found for Translation client.")
            return None
    except Exception as e:
        print(f"Error initializing Translation client: {e}")
        return None

def get_texttospeech_client(credentials_path: Optional[str]) -> Optional[texttospeech.TextToSpeechClient]:
    """Initializes and returns a Google Cloud Text-to-Speech client."""
    try:
        if credentials_path and os.path.exists(credentials_path):
            return texttospeech.TextToSpeechClient()
        else:
            print("Warning: Google Cloud credentials path not found for Text-to-Speech client.")
            return None
    except Exception as e:
        print(f"Error initializing Text-to-Speech client: {e}")
        return None

def get_speech_client(credentials_path: Optional[str]) -> Optional[speech.SpeechClient]:
    """Initializes and returns a Google Cloud Speech-to-Text client."""
    try:
        if credentials_path and os.path.exists(credentials_path):
            return speech.SpeechClient()
        else:
            print("Warning: Google Cloud credentials path not found for Speech-to-Text client.")
            return None
    except Exception as e:
        print(f"Error initializing Speech-to-Text client: {e}")
        return None


def translate_text(
    translator_client: Optional[translate.TranslationServiceClient],
    text: Optional[str],
    target_language_code: str,
    source_language_code: str,
    project_id: str
) -> Optional[str]:
    """Translates text into the target language."""
    if not translator_client or not text or not text.strip():
        return text # Return original if no client, no text, or empty text

    # Handle cases where source and target are the same
    if source_language_code == target_language_code:
        return text

    parent = f"projects/{project_id}/locations/global"

    try:
        response = translator_client.translate_text(
            request={
                "parent": parent,
                "contents": [text],
                "mime_type": "text/plain",
                "source_language_code": source_language_code,
                "target_language_code": target_language_code,
            }
        )
        return response.translations[0].translated_text
    except Exception as e:
        print(f"Error translating text from {source_language_code} to {target_language_code}: {e}")
        return text # Return original text on error


def get_supported_languages(
    translator_client: Optional[translate.TranslationServiceClient],
    project_id: str,
    display_language_code: str = "en",
    allowed_langs: Optional[List[str]] = None
) -> Dict[str, str]:
    """Returns a dictionary of supported language codes and their display names."""
    if not translator_client:
        print("Translator client not available to fetch supported languages.")
        return {}

    parent = f"projects/{project_id}/locations/global"
    languages: Dict[str, str] = {}
    try:
        response = translator_client.get_supported_languages(
            request={"parent": parent, "display_language_code": display_language_code}
        )
        for lang in response.languages:
            if allowed_langs is None or lang.language_code in allowed_langs:
                languages[lang.language_code] = lang.display_name
        return languages
    except Exception as e:
        print(f"Error fetching supported languages: {e}")
        return {}


def transcribe_audio(
    speech_client: Optional[speech.SpeechClient],
    audio_bytes: bytes,
    language_code: str,
    sample_rate_hertz: int, # <--- NEW PARAMETER
    channels: int          # <--- NEW PARAMETER
) -> Optional[str]:
    """
    Transcribes audio using Google Cloud Speech-to-Text.

    Args:
        speech_client: Initialized Google Cloud SpeechClient.
        audio_bytes: The audio content as bytes (e.g., from st_audiorecorder).
        language_code: The language code of the speech (e.g., 'en', 'hi', 'ta', 'te').
        sample_rate_hertz: The exact sample rate of the audio (e.g., 48000, 16000).
        channels: The number of audio channels (1 for mono, 2 for stereo).

    Returns:
        The transcribed text string, or None if transcription fails.
    """
    if not speech_client:
        print("Error: Speech client not initialized for transcription.")
        return None

    # Map generic language codes to regional codes preferred by Google STT
    regional_language_code_map = {
        'en': 'en-US',
        'hi': 'hi-IN',
        'ta': 'ta-IN',
        'te': 'te-IN',
    }
    stt_lang_code = regional_language_code_map.get(language_code, language_code)


    audio = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        # pydub exports to LINEAR16 (PCM signed 16-bit little-endian) by default
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate_hertz, # <--- USE DYNAMIC VALUE
        language_code=stt_lang_code,
        audio_channel_count=channels,        # <--- USE DYNAMIC VALUE
        enable_automatic_punctuation=True,
    )

    try:
        # Perform synchronous speech recognition
        response = speech_client.recognize(config=config, audio=audio)

        if response.results:
            # Return the transcript of the first alternative of the first result
            return response.results[0].alternatives[0].transcript
        else:
            print("DEBUG: No transcription results found from Speech-to-Text API.")
            return None
    except Exception as e:
        print(f"Error transcribing audio with Google Cloud Speech-to-Text: {e}")
        return None