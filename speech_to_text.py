# stt_module.py
import queue
import asyncio
import time
import sys
import av # For audio frame processing from streamlit-webrtc
import numpy as np # Import numpy for array manipulation

from google.cloud import speech
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, AudioProcessorBase

# --- Constants for STT (re-define or pass from main) ---
# Google Cloud Speech-to-Text typically expects 16kHz sample rate
SAMPLE_RATE = 16000
# CHUNK_SIZE is not directly used for WebRTC audio processing, but useful for general context.
# CHUNK_SIZE = int(SAMPLE_RATE / 10) # 100ms chunks

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
        # Convert audio frame to a NumPy array (often float32 by default from WebRTC)
        audio_array = frame.to_ndarray()

        # Step 1: Handle stereo to mono conversion if necessary
        # The shape of audio_array for stereo could be (channels, samples) or (samples, channels).
        # pyAV's to_ndarray() typically produces (channels, samples) or (samples,) for mono.
        # Let's assume (channels, samples) for stereo or (samples,) for mono.
        if audio_array.ndim > 1 and audio_array.shape[0] == 2: # Check if it's stereo (2 channels)
            audio_array = audio_array.mean(axis=0) # Downmix to mono by averaging channels

        # Step 2: Ensure correct sample rate (resampling)
        # It's crucial that the `sample_rate_hertz` in RecognitionConfig matches the actual audio.
        # If the incoming frame.sample_rate is not 16000, we need to resample.
        if frame.sample_rate != SAMPLE_RATE:
            try:
                # Using scipy.signal.resample for resampling. Requires `pip install scipy`.
                # Note: `resampy` is often better quality for audio, but `scipy` is commonly available.
                from scipy.signal import resample
                num_samples_resampled = int(len(audio_array) * SAMPLE_RATE / frame.sample_rate)
                audio_array = resample(audio_array, num_samples_resampled)
                # print(f"STT Module: Resampled audio from {frame.sample_rate}Hz to {SAMPLE_RATE}Hz")
            except ImportError:
                print("Warning: scipy not installed. Cannot resample audio. Ensure browser's sample rate is 16kHz or install scipy.")
                # If scipy not available, continue with original sample rate but log warning
                # Google STT might still work but might be less accurate or throw errors.
            except Exception as e:
                print(f"Error during audio resampling: {e}")
                # Continue with original array, but STT might fail if sample rate mismatch is severe.

        # Step 3: Convert to 16-bit signed integer (LINEAR16/s16)
        # Scale float values from -1.0 to 1.0 to the int16 range (-32768 to 32767)
        audio_array = (audio_array * 32767).astype(np.int16)

        # Convert the NumPy array to raw bytes
        raw_bytes = audio_array.tobytes()

        # Put the processed audio chunk into the queue
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
        sample_rate_hertz=SAMPLE_RATE, # Ensure this matches processed audio
        language_code=selected_language,
        max_alternatives=1,
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    audio_generator = generate_audio_chunks()
    requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)

    try:
        # This is where the actual streaming recognition starts.
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
                # After a final result, the current STT session usually completes.
                # The generate_audio_chunks loop will also stop if IS_LISTENING becomes False.
                break # Exit the response loop after a final result

            else:
                # Update Streamlit UI with interim results
                if TRANSCRIPT_PLACEHOLDER:
                    # Use f-string for better formatting in markdown
                    TRANSCRIPT_PLACEHOLDER.markdown(f"**Interim:** *{transcript}*")
                await asyncio.sleep(0.01) # Small delay to yield control for UI updates

    except Exception as e:
        print(f"STT Module: Streaming transcription error: {e}")
    finally:
        IS_LISTENING = False # Ensure flag is reset on exit
        # Clear any remaining audio chunks in the queue to prevent old data
        # from affecting the next transcription session.
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

    # Set the global placeholders/callbacks which will be used by transcribe_realtime_google_cloud
    TRANSCRIPT_PLACEHOLDER = interim_display_placeholder
    FINAL_TRANSCRIPT_CALLBACK = final_transcript_callback

    # Ensure audio queue is clear before starting a new session to avoid old audio
    while not AUDIO_CHUNKS_QUEUE.empty():
        try:
            AUDIO_CHUNKS_QUEUE.get_nowait()
        except queue.Empty:
            break

    # Initialize webrtc_streamer to get audio from the user's microphone
    webrtc_ctx = webrtc_streamer(
        key="stt_webrtc_stream", # Unique key for the component to prevent re-creation issues
        mode=WebRtcMode.SENDONLY, # We only need to send audio to the server, not receive
        audio_html_attrs={"autoPlay": True, "controls": False}, # Autoplay microphone, hide controls
        media_stream_constraints={"video": False, "audio": True}, # Request only audio
        audio_processor_factory=WebRtcAudioProcessor, # Our custom processor for audio formatting
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]} # STUN server for NAT traversal
        ),
        async_processing=True # Allow the audio processor to run asynchronously
    )

    # If the WebRTC stream is playing and the STT task isn't already active
    if webrtc_ctx.state.playing and not IS_LISTENING:
        # Start the transcription task in the background using asyncio
        asyncio.create_task(
            transcribe_realtime_google_cloud(speech_client, selected_language)
        )
        print("STT Module: WebRTC stream started and transcription task initiated.")
    elif not webrtc_ctx.state.playing and IS_LISTENING:
        # This block handles scenarios where the browser stops the microphone
        # (e.g., user revokes permission, tab closed) while the STT task is running.
        # It ensures that our internal IS_LISTENING flag is correctly reset.
        stop_webrtc_stream(webrtc_ctx) # Explicitly stop our internal logic
        print("STT Module: WebRTC stream detected as stopped externally; STT task terminated.")

    return webrtc_ctx

def stop_webrtc_stream(webrtc_ctx):
    """
    Stops the streamlit-webrtc audio stream and transcription process.
    """
    global IS_LISTENING

    # If the webrtc context exists and is playing, stop it.
    if webrtc_ctx and webrtc_ctx.state.playing:
        webrtc_ctx.stop()
        print("STT Module: WebRTC context stopped.")

    # If the STT task is still marked as listening, signal it to stop.
    if IS_LISTENING:
        IS_LISTENING = False # Signal the transcriber to stop
        # Put a sentinel value (None) into the queue to unblock `queue.get(timeout=0.1)`
        # in `generate_audio_chunks` immediately, allowing it to exit.
        AUDIO_CHUNKS_QUEUE.put(None)
        print("STT Module: Signaled transcription task to stop.")