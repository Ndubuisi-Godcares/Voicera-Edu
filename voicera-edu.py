import streamlit as st
import tempfile
import os
import speech_recognition as sr
from gtts import gTTS
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Cohere  # Keep using community LLM package
from langchain.chains.question_answering import load_qa_chain
from pydub import AudioSegment

# Load Cohere API key
cohere_api_key = st.secrets["cohere_api_key"]

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
    with st.spinner("Processing PDF..."):
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            if page.extract_text():
                doc_text += page.extract_text().strip() + "\n"
        
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(doc_text)
        
        embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key, model="embed-english-v3.0")
        docsearch = FAISS.from_texts(texts, embeddings)
        
        # Fix: Create Cohere LLM with correct parameters
        llm = Cohere(cohere_api_key=cohere_api_key, temperature=0.3)
        chain = load_qa_chain(llm, chain_type="stuff")
        
        st.success(f"PDF processed with {len(texts)} text chunks")

# Step 2: Voice or Text Input
query = ""
audio_bytes = st.audio_input("🎤 Ask your question by voice")

if audio_bytes is not None:
    try:
        # Create temp directory if it doesn't exist
        temp_dir = tempfile.mkdtemp()
        
        # Save the audio bytes to a temporary file
        temp_audio_path = os.path.join(temp_dir, "input.webm")
        with open(temp_audio_path, "wb") as f:
            f.write(audio_bytes.getvalue())
        
        # Convert to WAV using pydub
        try:
            audio = AudioSegment.from_file(temp_audio_path)
            wav_path = os.path.join(temp_dir, "input.wav")
            audio.export(wav_path, format="wav")
            
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
            query = recognizer.recognize_google(audio_data)
            st.success(f"You asked: {query}")
        except Exception as e:
            st.error(f"Audio conversion failed: {e}")
            st.info("Make sure ffmpeg is installed in your Streamlit environment. Try adding it to your requirements.txt")
            
        # Clean up temp files
        for file_path in [temp_audio_path, wav_path]:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.rmdir(temp_dir)
    except Exception as e:
        st.error(f"Voice recognition failed: {e}")

query = st.text_input("Or type your question here:", value=query)

# Step 3: Generate Answer
if query and docsearch and chain:
    with st.spinner("Thinking like a teacher..."):
        docs = docsearch.similarity_search(query)
        try:
            # Try the new invoke method first
            answer = chain.invoke({"input_documents": docs, "question": query})
            if isinstance(answer, dict) and "output_text" in answer:
                answer = answer["output_text"]
        except (AttributeError, TypeError):
            # Fall back to the run method if invoke fails
            answer = chain.run(input_documents=docs, question=query)
    
    st.markdown("**Answer:** " + answer)
    
    # Text-to-Speech
    with st.spinner("Generating audio response..."):
        tts = gTTS(text=answer, lang='en')
        audio_path = os.path.join(tempfile.gettempdir(), "answer.mp3")
        tts.save(audio_path)
        
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            # Auto-play the audio
            st.audio(audio_bytes, format="audio/mp3", start_time=0)
            
            # Add JavaScript to auto-play the audio
            st.markdown("""
            <script>
                document.addEventListener('DOMContentLoaded', (event) => {
                    const audioElements = document.querySelectorAll('audio');
                    audioElements.forEach((audio) => {
                        audio.play();
                    });
                });
            </script>
            """, unsafe_allow_html=True)
        
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

st.caption("Made with ❤️ for better learning experiences.")