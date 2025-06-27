# stt_module.py
import queue
import asyncio
import time
import sys
import av # For audio frame processing from streamlit-webrtc
import numpy as np
from scipy.signal import resample
from typing import cast # <--- IMPORTANT: Added for explicit type hinting to Pylance

from google.cloud import speech
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, AudioProcessorBase

# --- Constants for STT ---
SAMPLE_RATE = 16000 # Google Cloud Speech-to-Text typically expects 16kHz

# --- Global State for STT Module ---
AUDIO_CHUNKS_QUEUE = queue.Queue()
IS_LISTENING = False # Control flag for the transcription loop
TRANSCRIPT_PLACEHOLDER = None # Placeholder for interim results in Streamlit
FINAL_TRANSCRIPT_CALLBACK = None # Callback function to send final transcript to main.py

# --- Audio Processor Class for streamlit-webrtc ---
class WebRtcAudioProcessor(AudioProcessorBase):
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        """
        Receives audio frames from the browser, converts them to 16-bit PCM mono bytes
        at the target SAMPLE_RATE, and puts them into the global queue for the STT API.
        """
        try:
            # 1. Convert AV frame to NumPy array.
            # Use 'cast' to explicitly tell Pylance that the result IS a numpy.ndarray.
            # This helps Pylance's static analysis bypass its uncertainty.
            raw_audio_data = cast(np.ndarray, frame.to_ndarray())

            # 2. Ensure the array is float64 for precise intermediate processing.
            audio_array: np.ndarray = raw_audio_data.astype(np.float64)

        except Exception as e:
            print(f"Error converting AV frame to numpy array: {e}")
            # If conversion fails, return the original frame to keep the stream alive
            # but do not queue any audio.
            return frame

        # Step 1: Handle multi-channel (e.g., stereo) to mono conversion
        # 'to_ndarray()' for multi-channel usually gives (channels, samples).
        # For mono, it's typically (samples,).
        if audio_array.ndim > 1:
            if audio_array.shape[0] > 1: # Check if the first dimension is number of channels
                audio_array = audio_array.mean(axis=0) # Downmix to mono by averaging channels
            # If after mean, it's still (1, samples), flatten it to (samples,)
            if audio_array.ndim > 1:
                audio_array = audio_array.flatten()

        # Step 2: Resample to target SAMPLE_RATE if necessary
        if frame.sample_rate != SAMPLE_RATE:
            try:
                num_samples_resampled = int(len(audio_array) * SAMPLE_RATE / frame.sample_rate)
                audio_array = resample(audio_array, num_samples_resampled) #type: ignore
                # print(f"STT Module: Resampled audio from {frame.sample_rate}Hz to {SAMPLE_RATE}Hz") # Uncomment for debugging
            except ImportError:
                print("Warning: scipy not installed. Cannot resample audio. Ensure browser's sample rate is 16kHz or install scipy (`pip install scipy`).")
            except Exception as e:
                print(f"Error during audio resampling from {frame.sample_rate}Hz to {SAMPLE_RATE}Hz: {e}. STT accuracy may be affected.")


        # Step 3: Convert to 16-bit signed integer (LINEAR16/s16)
        # Scale float values (expected range -1.0 to 1.0) to the int16 range (-32768 to 32767)
        # np.iinfo(np.int16).max gives 32767
        scaled_audio_array = audio_array * np.iinfo(np.int16).max

        # Explicitly clip values to ensure they are within the valid int16 range
        scaled_audio_array = np.clip(scaled_audio_array, np.iinfo(np.int16).min, np.iinfo(np.int16).max)

        # Final conversion to 16-bit integer type
        final_audio_array: np.ndarray = scaled_audio_array.astype(np.int16)

        # Convert the final 16-bit NumPy array to raw bytes
        raw_bytes = final_audio_array.tobytes() # <--- THIS IS CRITICAL: Use final_audio_array here

        # Put the processed audio chunk into the global queue
        AUDIO_CHUNKS_QUEUE.put(raw_bytes)

        return frame # Return the original frame to keep the WebRTC stream alive

# --- Generator for Google Cloud STT ---
def generate_audio_chunks():
    """
    Generator that yields audio chunks from the queue for the STT API.
    """
    global IS_LISTENING
    while IS_LISTENING:
        try:
            # Use a timeout to allow the loop to check IS_LISTENING periodically
            chunk = AUDIO_CHUNKS_QUEUE.get(timeout=0.1)
            if chunk is None: # Sentinel value to stop the generator
                break
            yield chunk
        except queue.Empty:
            # If queue is empty, continue looping to check IS_LISTENING
            continue
    print("STT Module: Audio chunk generator stopped.")


# --- Main Real-time Transcription Function ---
async def transcribe_realtime_google_cloud(
    speech_client: speech.SpeechClient,
    selected_language: str,
):
    """
    Handles the continuous streaming of audio to Google Cloud Speech-to-Text
    and processes the responses.
    """
    global IS_LISTENING, TRANSCRIPT_PLACEHOLDER, FINAL_TRANSCRIPT_CALLBACK

    if not speech_client:
        print("STT Module: Speech-to-Text client not initialized. Cannot transcribe.")
        IS_LISTENING = False
        return

    print(f"STT Module: Starting real-time transcription for language: {selected_language}")
    IS_LISTENING = True # Set listening flag to true

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE, # Ensure this matches processed audio (16kHz)
        language_code=selected_language,
        max_alternatives=1,
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    audio_generator = generate_audio_chunks()
    requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)

    try:
        responses = speech_client.streaming_recognize(streaming_config, requests)

        for response in responses:
            if not IS_LISTENING: # Check flag to stop processing responses immediately
                break

            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript

            if result.is_final:
                print(f"STT Module: Final Transcript: {transcript}")
                if FINAL_TRANSCRIPT_CALLBACK:
                    FINAL_TRANSCRIPT_CALLBACK(transcript) # Call the callback in main.py
                if TRANSCRIPT_PLACEHOLDER:
                    TRANSCRIPT_PLACEHOLDER.empty() # Clear interim display after final
                break # Exit the response loop after a final result

            else:
                # Update Streamlit UI with interim results
                if TRANSCRIPT_PLACEHOLDER:
                    TRANSCRIPT_PLACEHOLDER.markdown(f"**Interim:** *{transcript}*")
                await asyncio.sleep(0.01) # Small delay to yield control for UI updates

    except Exception as e:
        print(f"STT Module: Streaming transcription error: {e}")
    finally:
        IS_LISTENING = False # Ensure flag is reset on exit
        # Clear any remaining audio chunks in the queue to prevent old data
        while not AUDIO_CHUNKS_QUEUE.empty():
            try:
                AUDIO_CHUNKS_QUEUE.get_nowait()
            except queue.Empty:
                break
        print("STT Module: Real-time transcription task finished.")


def start_webrtc_stream(
    speech_client: speech.SpeechClient,
    selected_language: str,
    interim_display_placeholder, # Pass the st.empty() object from main.py
    final_transcript_callback, # Pass the function to call with final transcript in main.py
):
    """
    Initializes and starts the streamlit-webrtc audio stream.
    Returns the webrtc_context.
    """
    global IS_LISTENING, TRANSCRIPT_PLACEHOLDER, FINAL_TRANSCRIPT_CALLBACK

    TRANSCRIPT_PLACEHOLDER = interim_display_placeholder
    FINAL_TRANSCRIPT_CALLBACK = final_transcript_callback

    # Ensure audio queue is clear before starting a new session
    while not AUDIO_CHUNKS_QUEUE.empty():
        try:
            AUDIO_CHUNKS_QUEUE.get_nowait()
        except queue.Empty:
            break

    # Initialize webrtc_streamer to get audio from the user's microphone
    webrtc_ctx = webrtc_streamer(
        key="stt_webrtc_stream", # Unique key for the component
        mode=WebRtcMode.SENDONLY, # Only send audio
        audio_html_attrs={"autoPlay": True, "controls": False}, # Autoplay mic, hide controls
        media_stream_constraints={"video": False, "audio": True}, # Request only audio
        audio_processor_factory=WebRtcAudioProcessor, # Our custom processor
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]} # STUN server
        ),
        async_processing=True # Allow async processing
    )

    # If WebRTC stream is playing and STT task isn't already active
    if webrtc_ctx.state.playing and not IS_LISTENING:
        asyncio.create_task(
            transcribe_realtime_google_cloud(speech_client, selected_language)
        )
        print("STT Module: WebRTC stream started and transcription task initiated.")
    elif not webrtc_ctx.state.playing and IS_LISTENING:
        # Handle cases where browser stops mic or component stops externally
        stop_webrtc_stream(webrtc_ctx)
        print("STT Module: WebRTC stream detected as stopped externally; STT task terminated.")

    return webrtc_ctx

def stop_webrtc_stream(webrtc_ctx):
    """
    Stops the streamlit-webrtc audio stream and transcription process.
    """
    global IS_LISTENING

    if webrtc_ctx and webrtc_ctx.state.playing:
        webrtc_ctx.stop()
        print("STT Module: WebRTC context stopped.")

    if IS_LISTENING:
        IS_LISTENING = False # Signal the transcriber to stop
        AUDIO_CHUNKS_QUEUE.put(None) # Put a sentinel value to unblock queue.get()
        print("STT Module: Signaled transcription task to stop.")