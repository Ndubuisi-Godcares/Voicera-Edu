import streamlit as st
import os
import tempfile
import speech_recognition as sr
from gtts import gTTS
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Cohere
from langchain.chains.question_answering import load_qa_chain
from pydub import AudioSegment
from datetime import datetime
import streamlit.components.v1 as components
import uuid

# Load Cohere API key
cohere_api_key = st.secrets["cohere_api_key"]

# Page config
st.set_page_config(
    page_title="Voicera - Conversational AI for Education",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    body { font-family: 'Segoe UI', sans-serif; }
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
</style>
""", unsafe_allow_html=True)

# Header
st.title("🤖 Voicera - Conversational AI for Education")
st.caption("Ask a question by voice or text from your PDF folder knowledge base.")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "audio_responses" not in st.session_state:
    st.session_state.audio_responses = {}

# Load documents from 'pdf_docs' folder
doc_text = ""
pdf_folder = "SSC_Syllabus"
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

if pdf_files:
    with st.spinner("Loading and processing PDFs..."):
        try:
            for pdf_file in pdf_files:
                reader = PdfReader(os.path.join(pdf_folder, pdf_file))
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        doc_text += text.strip() + "\n"

            splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
            texts = splitter.split_text(doc_text)
            embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key, model="embed-english-v3.0")
            docsearch = FAISS.from_texts(texts, embeddings)
            llm = Cohere(cohere_api_key=cohere_api_key, temperature=0.3)
            chain = load_qa_chain(llm, chain_type="stuff")
            st.session_state.document_processed = True
            st.success(f"{len(pdf_files)} PDFs processed ({len(texts)} sections)")
        except Exception as e:
            st.error(f"Failed to process PDFs: {str(e)}")
else:
    st.warning("No PDF files found in the `pdf_docs` folder.")

# Sidebar preview
with st.sidebar:
    st.header("📁 PDF Folder Content")
    if pdf_files:
        st.write("**Files:**")
        for f in pdf_files:
            st.write(f"- {f}")
        st.markdown("**Content Preview:**")
        st.markdown(f'<div class="document-content">{doc_text[:3000]}</div>', unsafe_allow_html=True)
        st.download_button("📅 Download Combined Text", doc_text, "combined_content.txt")
    else:
        st.info("Add PDFs to `pdf_docs/` folder to enable features.")

# Question Input
with st.expander("💬 Ask Your Question"):
    query = ""

    # Voice input
    audio_bytes = st.audio_input("Speak your question:")
    if audio_bytes:
        try:
            temp_dir = tempfile.mkdtemp()
            webm_path = os.path.join(temp_dir, "input.webm")
            with open(webm_path, "wb") as f:
                f.write(audio_bytes.getvalue())
            audio = AudioSegment.from_file(webm_path)
            wav_path = os.path.join(temp_dir, "input.wav")
            audio.export(wav_path, format="wav")
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
            query = recognizer.recognize_google(audio_data)
            st.session_state.chat_history.append({"type": "user", "content": query, "timestamp": datetime.now().strftime("%H:%M")})
            os.remove(webm_path)
            os.remove(wav_path)
            os.rmdir(temp_dir)
        except Exception as e:
            st.error(f"Speech recognition failed: {str(e)}")

    # Text input
    query = st.text_input("Or type your question:", value=query)

# Answering
if query and st.session_state.document_processed:
    if not any(m['content'] == query for m in st.session_state.chat_history if m['type'] == 'user'):
        st.session_state.chat_history.append({"type": "user", "content": query, "timestamp": datetime.now().strftime("%H:%M")})
    with st.spinner("Answering your question..."):
        try:
            docs = docsearch.similarity_search(query)
            result = chain.invoke({"input_documents": docs, "question": query})
            answer = result.get("output_text", "I couldn't find a good answer.")
            st.session_state.chat_history.append({"type": "bot", "content": answer, "timestamp": datetime.now().strftime("%H:%M")})

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

# Chat History Display
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

# Chat Summary
def generate_summary(history):
    if not history:
        return "No conversation yet."
    return "\n".join(f"{'User' if h['type'] == 'user' else 'Bot'} ({h['timestamp']}): {h['content']}" for h in history)

if st.button("📌 Summarize Chat"):
    summary = generate_summary(st.session_state.chat_history)
    components.html(f"""
    <div style='position:fixed;bottom:20px;right:20px;width:350px;max-height:300px;background:#fff;padding:1rem;border:1px solid #ccc;border-radius:12px;overflow:auto;z-index:9999;'>
        <h4>🧠 Chat Summary</h4>
        <pre style='white-space: pre-wrap;font-size:13px;'>{summary}</pre>
        <button onclick=\"this.parentElement.style.display='none'\" style='margin-top:10px;background:#4f46e5;color:#fff;border:none;padding:0.5rem 1rem;border-radius:6px;'>Close</button>
    </div>
    """, height=400)
