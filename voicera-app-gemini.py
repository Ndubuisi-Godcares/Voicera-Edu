import streamlit as st
import tempfile
import os
import speech_recognition as sr
from gtts import gTTS
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from pydub import AudioSegment
from datetime import datetime
import base64
import uuid
import google.generativeai as genai
from io import BytesIO

# Load Gemini API key
genai.configure(api_key=st.secrets["gemini_api_key"])

# Set page config
st.set_page_config(
    page_title="Voicera - Conversational AI for Education",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    body {
        font-family: 'Segoe UI', sans-serif;
    }
    .upload-container {
        border: 2px dashed #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        background: #f9fafb;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-bubble {
        max-width: 80%;
        padding: 0.75rem 1rem;
        border-radius: 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
        line-height: 1.4;
        color: #111827;
    }
    .user-bubble {
        background-color: #e0e7ff;
        align-self: flex-end;
        border-bottom-right-radius: 0;
    }
    .bot-bubble {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        box-shadow: 0px 1px 4px rgba(0,0,0,0.1);
        border-bottom-left-radius: 0;
    }
    .timestamp {
        font-size: 0.7rem;
        color: #9ca3af;
        margin-top: 0.25rem;
    }
    .document-content {
        max-height: 250px;
        overflow-y: auto;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.75rem;
        font-family: monospace;
        white-space: pre-wrap;
        color: #111827;
    }
    .card {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        margin: 15px 0;
        padding: 20px;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache document processing
@st.cache_resource(show_spinner="Processing document, please wait...")
def process_document(file_bytes):
    reader = PdfReader(BytesIO(file_bytes))
    doc_text = ""
    max_pages = 20
    for i, page in enumerate(reader.pages):
        if i >= max_pages:
            break
        text = page.extract_text()
        if text:
            doc_text += text.strip() + "\n"

    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_text(doc_text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=st.secrets["gemini_api_key"]
    )
    docsearch = FAISS.from_texts(texts, embeddings)

    return doc_text, texts, docsearch

# App Header
st.title("🤖 Voicera - Conversational AI for Education")
st.caption("Upload a textbook or syllabus (PDF), then ask a question by voice or text to get an instant spoken response.")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "audio_responses" not in st.session_state:
    st.session_state.audio_responses = {}

# Upload
with st.expander("📄 Upload Learning Materials"):
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

# Process uploaded file
if uploaded_file:
    file_bytes = uploaded_file.read()
    with st.spinner("Processing document..."):
        try:
            doc_text, texts, docsearch = process_document(file_bytes)
            st.session_state.document_processed = True
            st.success(f"Document processed ({len(texts)} sections)")
        except Exception as e:
            st.error(f"Failed to process: {str(e)}")
else:
    doc_text, texts, docsearch = "", [], None

# Sidebar tools
with st.sidebar:
    st.header("📁 Document Tools")
    if uploaded_file and st.session_state.document_processed:
        st.write(f"**Name:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
        st.write(f"**Sections:** {len(texts)}")
        st.markdown("**Content Preview:**")
        st.markdown(f'<div class="document-content">{doc_text}</div>', unsafe_allow_html=True)
        st.download_button("📅 Download Text", doc_text, f"{uploaded_file.name}_content.txt")
    else:
        st.info("Upload a document to enable tools")

# Ask Question
with st.expander("💬 Ask Your Question"):
    query = ""

    # Voice input
    audio_bytes = st.audio_input("Speak your question:")
    if audio_bytes:
        try:
            temp_dir = tempfile.mkdtemp()
            webm_path = os.path.join(temp_dir, "input.webm")
            wav_path = os.path.join(temp_dir, "input.wav")
            with open(webm_path, "wb") as f:
                f.write(audio_bytes.getvalue())
            audio = AudioSegment.from_file(webm_path)
            audio.export(wav_path, format="wav")
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
            query = recognizer.recognize_google(audio_data)
            st.session_state.chat_history.append({
                "type": "user", "content": query,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            os.remove(webm_path)
            os.remove(wav_path)
            os.rmdir(temp_dir)
        except Exception as e:
            st.error(f"Speech recognition failed: {str(e)}")

    # Text fallback input
    query = st.text_input("Or type your question:", value=query)

# Answer the query
if query and st.session_state.document_processed:
    if not any(m['content'] == query for m in st.session_state.chat_history if m['type'] == 'user'):
        st.session_state.chat_history.append({
            "type": "user", "content": query,
            "timestamp": datetime.now().strftime("%H:%M")
        })
    with st.spinner("Answering your question..."):
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
            chain = load_qa_chain(llm, chain_type="stuff")
            docs = docsearch.similarity_search(query)
            result = chain.invoke({"input_documents": docs, "question": query})
            answer = result.get("output_text", "I couldn't find a good answer.")
            st.session_state.chat_history.append({
                "type": "bot", "content": answer,
                "timestamp": datetime.now().strftime("%H:%M")
            })

            # TTS audio generation
            response_id = str(uuid.uuid4())
            tts = gTTS(text=answer, lang='en')
            temp_dir = tempfile.mkdtemp()
            audio_path = os.path.join(temp_dir, f"response_{response_id}.mp3")
            tts.save(audio_path)

            with open(audio_path, "rb") as audio_file:
                st.session_state.audio_responses[response_id] = audio_file.read()

            st.audio(st.session_state.audio_responses[response_id], format="audio/mp3")

            os.remove(audio_path)
            os.rmdir(temp_dir)

        except Exception as e:
            st.error(f"Response error: {str(e)}")

# Chat History
st.subheader("📜 Chat History")
if not st.session_state.chat_history:
    st.info("Your conversation will appear here.")
else:
    for msg in reversed(st.session_state.chat_history):
        bubble_class = "user-bubble" if msg['type'] == 'user' else "bot-bubble"
        st.markdown(f"""
        <div class='chat-bubble {bubble_class}'>
            {msg['content']}
            <div class='timestamp'>{msg['timestamp']}</div>
        </div>
        """, unsafe_allow_html=True)

# Summary popup
def generate_summary(history):
    return "\n".join(f"{'User' if h['type'] == 'user' else 'Bot'} ({h['timestamp']}): {h['content']}" for h in history) or "No conversation yet."

if st.button("📌 Summarize Chat"):
    summary = generate_summary(st.session_state.chat_history)
    st.text_area("Chat Summary:", summary, height=300)
