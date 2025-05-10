import streamlit as st
import tempfile
import os
import speech_recognition as sr
from gtts import gTTS
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import cohere  # Corrected import
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from pydub import AudioSegment

# Load Cohere API key
cohere_api_key = st.secrets["cohere_api_key"]

# Set page configuration with a custom layout
st.set_page_config(page_title="AI Teacher Voice Assistant", layout="wide")
st.title("🧑‍🏫 AI Teacher Voice Assistant")

# Custom CSS for the UI
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f4f7f6;
            color: #333;
        }
        .stButton>button {
            background-color: #6c63ff;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 10px 20px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #5c52e4;
        }
        .stTextInput>div>div>input {
            font-size: 18px;
        }
        .stTextInput {
            width: 100%;
            margin-bottom: 20px;
        }
        .stAudio>audio {
            border-radius: 8px;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
        }
        .description {
            font-size: 18px;
            text-align: center;
            color: #666;
        }
    </style>
""", unsafe_allow_html=True)

# Page header
st.markdown("""
    <div class="header">
        <h2>Upload a textbook or syllabus (PDF), ask a question using voice or text, and receive a spoken answer from your AI Teacher! 🎤📚</h2>
    </div>
""", unsafe_allow_html=True)

# Step 1: Upload PDF
uploaded_file = st.file_uploader("Upload your syllabus or textbook (PDF)", type=["pdf"], label_visibility="collapsed")
doc_text = ""
docsearch = None
chain = None

if uploaded_file:
    with st.spinner("Processing your document..."):
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            if page.extract_text():
                doc_text += page.extract_text().strip() + "\n"

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(doc_text)

        # Using the cohere API for embeddings
        embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key, model="embed-english-v3.0")
        docsearch = FAISS.from_texts(texts, embeddings)
        chain = load_qa_chain(cohere.Client(cohere_api_key), chain_type="stuff")
    st.success("Document processed successfully!")

# Step 2: Voice or Text Input
query = ""
st.markdown("<div class='description'>Ask your question using voice or text below:</div>", unsafe_allow_html=True)

# Button for voice input
voice_button = st.button("🎤 Ask using Voice")

if voice_button:
    st.info("Listening... Please speak now.")
    try:
        audio_bytes = st.audio_input("Click here to speak")

        if audio_bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
                temp_audio.write(audio_bytes.read())
                temp_audio_path = temp_audio.name

            # Convert to WAV using pydub
            audio = AudioSegment.from_file(temp_audio_path)
            wav_path = temp_audio_path.replace(".webm", ".wav")
            audio.export(wav_path, format="wav")

            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
            query = recognizer.recognize_google(audio_data)
            st.success(f"You asked: {query}")
            os.remove(temp_audio_path)
            os.remove(wav_path)
    except Exception as e:
        st.error(f"Voice recognition failed: {e}")

# Text input if no voice input
query = st.text_input("Or type your question here:", value=query, label_visibility="collapsed")

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

    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3")
    os.remove(audio_path)

# Animated Avatar
st.markdown("""
    <div style='text-align: center;'>
        <lottie-player src='https://assets2.lottiefiles.com/packages/lf20_mf2zpxsb.json' background='transparent' speed='1' style='width: 300px; height: 300px;' loop autoplay></lottie-player>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
""", unsafe_allow_html=True)

# Footer
st.caption("Made with ❤️ for better learning experiences.")
