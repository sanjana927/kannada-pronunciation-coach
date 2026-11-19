
import streamlit as st
import os
import random
import whisper
import tempfile
import re
import difflib

st.set_page_config(page_title="Kannada Pronunciation Coach")

st.title("Kannada Pronunciation Coach")
st.write("Practice speaking Kannada and improve your pronunciation!")

# Load Whisper model
@st.cache_resource
def load_whisper():
    return whisper.load_model("medium")


# Dataset location
DATA_FOLDER = os.path.expanduser("~/slr79")
TRANSCRIPT_FILE = os.path.join(DATA_FOLDER, "line_index_male.tsv")


# Load Kannada sentences and audio files
sentences = {}

with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split("\t", 1)

        if len(parts) == 2:
            audio_id, sentence = parts
            audio_path = os.path.join(DATA_FOLDER, audio_id + ".wav")

            if os.path.exists(audio_path):
                sentences[audio_id] = {
                    "text": sentence,
                    "audio": audio_path
                }


st.success(f"Loaded {len(sentences)} Kannada audio samples!")


if sentences:

    # Keep the same sentence while recording
    if "audio_id" not in st.session_state:
        st.session_state.audio_id = random.choice(list(sentences.keys()))

    selected = sentences[st.session_state.audio_id]

    # Display target sentence
    st.subheader("1. Your Kannada sentence")
    st.write(selected["text"])

    # Play reference audio
    st.subheader("2. Reference pronunciation")
    st.audio(selected["audio"], format="audio/wav")

    # Record user's voice
    st.subheader("3. Record your pronunciation")
    recording = st.audio_input("Record your voice")

    if recording:

        st.audio(recording)

        if st.button("Check my pronunciation"):

            with st.spinner("Whisper is listening..."):

                model = load_whisper()

                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as temp_file:

                    temp_file.write(recording.getvalue())
                    temp_path = temp_file.name

                try:

                    # Convert speech into Kannada text
                    result = model.transcribe(
                        temp_path,
                        language="kn",
                        fp16=False
                    )

                    transcribed_text = result["text"]

                    st.subheader("4. What Whisper understood")
                    st.write(transcribed_text)

                    # Prepare both sentences for comparison
                    target_words = re.findall(
                        r'\w+', selected["text"].lower()
                    )

                    spoken_words = re.findall(
                        r'\w+', transcribed_text.lower()
                    )

                    # Calculate text similarity
                    similarity = difflib.SequenceMatcher(
                        None,
                        target_words,
                        spoken_words
                    ).ratio() * 100

                    st.subheader("5. Your Practice Feedback")

                    st.metric(
                        "Approximate text similarity",
                        f"{similarity:.1f}%"
                    )

                    if similarity >= 80:
                        st.success(
                            "Great job! The recognized words are close to the target."
                        )

                    elif similarity >= 50:
                        st.warning(
                            "Good effort! Keep practising this sentence."
                        )

                    else:
                        st.info(
                            "Keep trying! Speak slowly and clearly."
                        )

                    # Display comparison
                    st.write("**Expected Kannada sentence:**")
                    st.write(selected["text"])

                    st.write("**Whisper recognized:**")
                    st.write(transcribed_text)

                    st.caption(
                        "Note: This score measures approximate text similarity. "
                        "It does not directly measure pronunciation accuracy."
                    )

                except Exception as e:
                    st.error(f"Error: {e}")

                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    # Select another sentence
    if st.button("Try another sentence"):
        st.session_state.audio_id = random.choice(
            list(sentences.keys())
        )
        st.rerun()

else:
    st.error("No matching audio files were found.")
