import io
from datetime import datetime
import speech_recognition as sr
import streamlit as st
from gtts import gTTS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Group 4 Voice Assistant", page_icon="🎙️")

commands = [
    "what is the time", "tell me the current time", "time please", "what time is it now",
    "what is today's date", "tell me the date", "what day is today", "current date please",
    "hello assistant", "hi there", "good morning", "hey buddy",
    "who are you", "what can you do", "introduce yourself", "help me",
    "tell me a joke", "make me laugh", "say something funny", "joke please"
]
intents = (
    ["time"] * 4 +
    ["date"] * 4 +
    ["greeting"] * 4 +
    ["about"] * 4 +
    ["joke"] * 4
)

@st.cache_resource
def train_assistant():
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(commands)
    model = LogisticRegression().fit(X, intents)
    return vectorizer, model

vectorizer, model = train_assistant()

st.title("🎙️ Mini Voice Assistant (Group 4)")
st.write("Speak a command using your mic **or** type it below. The AI will understand and speak back!")

audio_value = st.audio_input("🎤 Speak your command (e.g., 'What is the time?')")
typed_command = st.text_input("⌨️ Or type your command here:")

user_command = ""

if audio_value is not None:
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_value) as source:
        audio_data = recognizer.record(source)
        try:
            user_command = recognizer.recognize_google(audio_data)
            st.info(f"**You said:** \"{user_command}\"")
        except sr.UnknownValueError:
            st.warning("Could not understand the audio clearly. Try speaking closer to the mic or type below.")
        except sr.RequestError:
            st.error("Speech recognition service is unavailable right now. Please use the text box below.")

if not user_command and typed_command.strip():
    if st.button("Run Typed Command"):
        user_command = typed_command.strip()

if user_command:
    vec = vectorizer.transform([user_command.lower()])
    probs = model.predict_proba(vec)[0]
    best_idx = probs.argmax()
    confidence = probs[best_idx]
    intent = model.classes_[best_idx]

    if confidence < 0.28:
        reply = "Sorry, I did not understand that command. Try asking for the time, date, or a joke."
        detected_label = "Unknown / Low Confidence"
    elif intent == "time":
        now = datetime.now().strftime("%I:%M %p")
        reply = f"The current time is {now}."
        detected_label = "Get Time"
    elif intent == "date":
        today = datetime.now().strftime("%B %d, %Y")
        reply = f"Today's date is {today}."
        detected_label = "Get Date"
    elif intent == "greeting":
        reply = "Hello! I am your Group 4 voice assistant. How can I help you today?"
        detected_label = "Greeting"
    elif intent == "about":
        reply = "I am G4 Voice Assistant"
        detected_label = "About Assistant"
    elif intent == "joke":
        reply = "Why do programmers prefer dark mode? Because light attracts bugs!"
        detected_label = "Tell Joke"

    st.success(f"**Detected Intent:** {detected_label} ({confidence:.0%} confidence)")
    st.write(f"**Assistant Reply:** {reply}")

    tts = gTTS(text=reply, lang="en")
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    st.audio(audio_buffer, format="audio/mp3", autoplay=True)

with st.expander("Limitations, Bias, and Ethics"):
    st.write(
        "- **Background Noise & Accents:** Speech-to-Text accuracy drops in noisy classrooms or with strong regional accents.\n"
        "- **Small Training Set:** Trained on only 20 sample phrases, so unfamiliar commands trigger a fallback.\n"
        "- **Language Bias:** Supports English only in this demo.\n"
        "- **Privacy:** Audio is only recorded when you click the mic button—no passive wake-word listening or data storage."
    )
