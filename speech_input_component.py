# speech_input_component.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av # 'av' library (installed via streamlit-webrtc)
from pydub import AudioSegment # 'pydub' library
import io
from typing import Optional, List

# Import transcription utility and SpeechClient
from trans_trials import transcribe_audio # Corrected to trans_trials
from google.cloud import speech

# Public STUN servers to help WebRTC establish peer-to-peer connections (important for NAT traversal)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def audio_input_widget(speech_client: Optional[speech.SpeechClient], current_language_code: str) -> Optional[str]:
    transcribed_text_output: Optional[str] = None

    st.subheader("Speak your Query in Real-time:")

    # --- Session State Management ---
    if 'recording_active' not in st.session_state:
        st.session_state.recording_active = False
    if 'audio_buffer' not in st.session_state:
        st.session_state.audio_buffer = []
    if 'last_transcribed_audio_hash' not in st.session_state:
        st.session_state.last_transcribed_audio_hash = None
    if 'webrtc_last_state_playing' not in st.session_state:
        st.session_state.webrtc_last_state_playing = False


    # --- WebRTC Streamer Setup ---
    webrtc_ctx = webrtc_streamer(
        key="audio_webrtc_recorder",
        mode=WebRtcMode.SENDONLY, # We only need to send audio from client (browser) to server (Streamlit)
        audio_receiver_size=256,   # Buffer size (adjust as needed, larger means more latency but less dropped frames)
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": False, "audio": True}, # Only request audio, no video
    )

    # Detect state changes to clear buffer if streamer just started playing
    if webrtc_ctx.state.playing and not st.session_state.webrtc_last_state_playing:
        st.session_state.audio_buffer = [] # Clear buffer on fresh connection/play
        st.session_state.recording_active = False # Reset recording state
        st.info("Microphone connected. Click 'Start Recording' to begin.")
    st.session_state.webrtc_last_state_playing = webrtc_ctx.state.playing


    # --- UI Buttons ---
    col1, col2 = st.columns([1, 1])

    with col1:
        start_button = st.button(
            "🔴 Start Recording",
            disabled=st.session_state.recording_active or not webrtc_ctx.state.playing
        )
    with col2:
        stop_button = st.button(
            "⏹ Stop Recording",
            disabled=not st.session_state.recording_active
        )

    # --- Handle Start Button Click ---
    if start_button:
        st.session_state.recording_active = True
        st.session_state.audio_buffer = [] # Clear buffer for a new recording
        st.info("Recording started... Please speak clearly.")
        st.session_state.last_transcribed_audio_hash = None # Allow new transcription
        st.rerun() # Rerun to update button states and messages

    # --- Handle Stop Button Click ---
    if stop_button:
        st.session_state.recording_active = False
        st.info("Recording stopped. Processing audio for transcription...")
        st.rerun() # Rerun to trigger the processing logic below

    # --- Frame Collection Loop (runs continuously while streamer is playing and recording_active is True) ---
    if webrtc_ctx.audio_receiver and webrtc_ctx.state.playing:
        # Collect frames when recording is active.
        try:
            for audio_frame in webrtc_ctx.audio_receiver.get_queued_frames():  # type: ignore # Suppress Pylance warning
                st.session_state.audio_buffer.append(audio_frame) # Store the av.AudioFrame objects
        except Exception as e:
            st.warning(f"Error collecting audio frames: {e}")

    # --- Audio Processing and Transcription Logic ---
    if not st.session_state.recording_active and len(st.session_state.audio_buffer) > 0 and st.session_state.last_transcribed_audio_hash == None:
        st.text("Assembling and processing recorded audio...")

        try:
            combined_audio = AudioSegment.empty()
            sample_width = 2 # Assuming 16-bit audio (common for mic input)
            frame_rate = None
            channels = None

            if len(st.session_state.audio_buffer) > 0:
                first_frame = st.session_state.audio_buffer[0]
                frame_rate = first_frame.sample_rate
                channels = first_frame.layout.channels
                st.text(f"Detected Audio Properties: Sample Rate={frame_rate} Hz, Channels={channels}")

            if frame_rate is None or channels is None:
                st.error("Could not determine audio frame properties. Please try again.")
                st.session_state.audio_buffer = []
                return None

            for frame in st.session_state.audio_buffer:
                audio_data_bytes = frame.planes[0].to_bytes() # Get raw bytes from the frame's first plane
                sound_segment = AudioSegment(
                    audio_data_bytes,
                    sample_width=sample_width,
                    frame_rate=frame_rate,
                    channels=channels
                )
                combined_audio += sound_segment

            # Export combined audio to WAV format in memory
            buffer_wav = io.BytesIO()
            combined_audio.export(buffer_wav, format="wav")
            audio_bytes_to_transcribe = buffer_wav.getvalue()

            # --- Perform Transcription ---
            current_audio_hash = hash(audio_bytes_to_transcribe)

            if speech_client:
                with st.spinner(f"Transcribing audio in {current_language_code}..."):
                    # PASS THE DYNAMIC SAMPLE_RATE_HERTZ AND CHANNELS HERE
                    transcribed_text = transcribe_audio(
                        speech_client,
                        audio_bytes_to_transcribe,
                        current_language_code,
                        sample_rate_hertz=frame_rate, # <--- NEW ARGUMENT
                        channels=channels             # <--- NEW ARGUMENT
                    )

                if transcribed_text:
                    st.success("Transcription successful!")
                    st.info(f"Transcribed: \"{transcribed_text}\"")
                    transcribed_text_output = transcribed_text
                    st.session_state.last_transcribed_audio_hash = current_audio_hash # Store hash of processed audio
                else:
                    st.error("Could not transcribe audio. Please try again or check microphone.")
            else:
                st.error("Speech-to-Text client is not initialized. Cannot process audio input.")

        except Exception as e:
            st.error(f"Error during audio processing or transcription: {e}")
            st.warning("Please ensure microphone access is allowed in your browser settings.")

        st.session_state.audio_buffer = [] # Clear buffer after processing
    elif not st.session_state.recording_active and len(st.session_state.audio_buffer) == 0 and st.session_state.last_transcribed_audio_hash is not None:
        pass
    elif not st.session_state.recording_active and len(st.session_state.audio_buffer) == 0:
        if webrtc_ctx.state.playing:
            st.warning("No audio frames were captured. Ensure you clicked 'Start Recording' and spoke.")


    # --- Display Status Messages ---
    if webrtc_ctx.state.playing and st.session_state.recording_active:
        st.write("🎙️ **Recording...** (Speak now)")
    elif webrtc_ctx.state.playing:
        st.info("Microphone is ready. Click 'Start Recording'.")
    elif not webrtc_ctx.state.playing:
        st.info("Waiting for microphone access...")

    return transcribed_text_output