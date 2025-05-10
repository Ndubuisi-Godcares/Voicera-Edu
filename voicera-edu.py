import streamlit as st
import tempfile
import os
import sounddevice as sd
import numpy as np
import wave
import speech_recognition as sr
from gtts import gTTS
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Cohere
from langchain.chains.question_answering import load_qa_chain

# Load Cohere API key from Streamlit secrets
cohere_api_key = st.secrets["cohere_api_key"]

# Streamlit page config
st.set_page_config(page_title="AI Teacher Voice Assistant", layout="centered")
st.title("🧑‍🏫 AI Teacher Voice Assistant")

st.markdown("""
Upload a textbook or syllabus (PDF), ask a question using voice or text, and receive a spoken answer from your AI Teacher! 🎤📚
""")

# Step 1: Upload PDF
uploaded_file = st.file_uploader("Upload a syllabus or textbook (PDF)", type=["pdf"])
doc_text = ""
docsearch = None
chain = None

if uploaded_file:
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        if page.extract_text():
            doc_text += page.extract_text().strip() + "\n"

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(doc_text)

    embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key, model="embed-english-v3.0")
    docsearch = FAISS.from_texts(texts, embeddings)
    chain = load_qa_chain(Cohere(cohere_api_key=cohere_api_key, temperature=0.3), chain_type="stuff")

# Step 2: Voice or Text Input
query = ""
voice_button = st.button("🎤 Ask using Voice")

if voice_button:
    try:
        # Record Audio using sounddevice
        st.info("Listening... Please speak now.")
        fs = 16000  # Sampling frequency
        duration = 5  # seconds

        # Record the audio
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()

        # Save to WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            wavefile = wave.open(fp.name, 'wb')
            wavefile.setnchannels(1)
            wavefile.setsampwidth(2)
            wavefile.setframerate(fs)
            wavefile.writeframes(audio_data.tobytes())
            audio_path = fp.name

        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        query = recognizer.recognize_google(audio)
        st.success(f"You asked: {query}")
        os.remove(audio_path)

    except Exception as e:
        st.warning("Voice input may not work in browser or Streamlit Cloud. Use the text box instead.")
        st.error(str(e))

query = st.text_input("Or type your question here:", value=query)

# Step 3: Generate Answer
if query and docsearch and chain:
    with st.spinner("Thinking like a teacher..."):
        docs = docsearch.similarity_search(query)
        answer = chain.run(input_documents=docs, question=query)

    st.markdown("**Answer:** " + answer)

    # Text-to-Speech
    tts = gTTS(text=answer, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        audio_path = fp.name

    audio_file = open(audio_path, "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3")
    os.remove(audio_path)

# Animated Avatar
st.markdown("""
<div style='text-align: center;'>
  <lottie-player src='https://assets2.lottiefiles.com/packages/lf20_mf2zpxsb.json' background='transparent' speed='1' style='width: 300px; height: 300px;' loop autoplay></lottie-player>
</div>
""", unsafe_allow_html=True)

# Lottie script loader
st.markdown("""
<script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
""", unsafe_allow_html=True)

st.caption("Made with ❤️ for better learning experiences.")
