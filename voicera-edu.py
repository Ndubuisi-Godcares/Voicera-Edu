import streamlit as st
import tempfile
import os
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

# Load Cohere API key
cohere_api_key = st.secrets["cohere_api_key"]

# Set page config (must be first Streamlit command)
st.set_page_config(
    page_title="Voicera - Conversational AI for Education",
    page_icon="🧑‍🏫",
    layout="centered"
)

# Custom CSS for professional styling with black text
st.markdown("""
<style>
    :root {
        --primary: #4f46e5;
        --primary-light: #6366f1;
        --secondary: #f9fafb;
        --text: #1f2937;
        --text-light: #6b7280;
        --border: #e5e7eb;
        --card-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    
    .upload-container {
        border: 2px dashed var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        background: var(--secondary);
    }
    
    .message-container {
        display: flex;
        margin-bottom: 12px;
    }
    
    .user-message {
        background-color: #e0e7ff;
        padding: 10px 14px;
        border-radius: 12px 12px 0 12px;
        margin-left: auto;
        max-width: 80%;
        color: #000000;  /* Black text */
    }
    
    .bot-message {
        background-color: white;
        padding: 10px 14px;
        border-radius: 12px 12px 12px 0;
        box-shadow: var(--card-shadow);
        max-width: 80%;
        color: #000000;  /* Black text */
    }
    
    .timestamp {
        font-size: 0.7rem;
        color: var(--text-light);
        margin-top: 4px;
    }
    
    .document-content {
        max-height: 300px;
        overflow-y: auto;
        padding: 10px;
        background: white;
        border-radius: 8px;
        border: 1px solid var(--border);
        margin-top: 10px;
        white-space: pre-wrap;
        font-family: monospace;
        color: #000000 !important;  /* Force black text */
    }
    
    .sidebar-section {
        margin-bottom: 1.5rem;
    }
    
    /* Force black text in all text elements */
    .stMarkdown, .stText, .stTextInput, .stTextArea {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.title("🎓 Voicera - Conversational AI for Education")
st.markdown(
    '<p style="color: #FFFFFF;">Upload a textbook or syllabus (PDF), then ask a question by <strong>voice</strong> or <strong>text</strong> to get an instant spoken response.</p>',
    unsafe_allow_html=True
)

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

# File upload section
with st.container():
    st.subheader("📄 Upload Learning Materials")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")

# Document processing
doc_text, docsearch, chain = "", None, None

if uploaded_file:
    with st.spinner("Analyzing document..."):
        try:
            # Extract text from PDF
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    doc_text += text.strip() + "\n"

            # Process document
            text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_text(doc_text)
            
            embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key, model="embed-english-v3.0")
            docsearch = FAISS.from_texts(texts, embeddings)
            
            llm = Cohere(cohere_api_key=cohere_api_key, temperature=0.3)
            chain = load_qa_chain(llm, chain_type="stuff")
            
            st.session_state.document_processed = True
            st.success(f"Document processed successfully ({len(texts)} sections)")
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")

# Document tools sidebar with black text
with st.sidebar:
    st.header("Document Tools")
    
    if uploaded_file:
        with st.expander("📋 Document Summary", expanded=True):
            st.markdown(f"""
            <div style="color: #FFFFFF;">
                <p><strong>File Name:</strong> {uploaded_file.name}</p>
                <p><strong>Size:</strong> {uploaded_file.size/1024:.1f} KB</p>
                <p><strong>Sections:</strong> {len(texts) if 'texts' in locals() else 0}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Document Content:**")
            st.markdown(f'<div class="document-content">{doc_text}</div>', unsafe_allow_html=True)
            
            st.download_button(
                "📥 Download Full Text",
                data=doc_text,
                file_name=f"{uploaded_file.name}_content.txt",
                mime="text/plain",
                use_container_width=True
            )
    else:
        st.info("Upload a document to access tools")

# Question input section
st.subheader("💬 Ask Your Question")
query = ""

# Voice input
audio_bytes = st.audio_input("Speak your question:", label_visibility="collapsed")

if audio_bytes is not None:
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
        
        # Add to chat history
        st.session_state.chat_history.append({
            "type": "user",
            "content": query,
            "timestamp": datetime.now().strftime("%H:%M")
        })

        os.remove(webm_path)
        os.remove(wav_path)
        os.rmdir(temp_dir)
    except Exception as e:
        st.error(f"Voice recognition failed: {str(e)}")

# Text input
query = st.text_input(
    "Type your question here:", 
    value=query, 
    label_visibility="collapsed"
)

# Answer questions
if query and docsearch and chain:
    if not any(msg['content'] == query and msg['type'] == 'user' for msg in st.session_state.chat_history):
        st.session_state.chat_history.append({
            "type": "user",
            "content": query,
            "timestamp": datetime.now().strftime("%H:%M")
        })
    
    with st.spinner("Generating response..."):
        try:
            docs = docsearch.similarity_search(query)
            
            if not docs:
                answer = "I couldn't find relevant information in the document."
            else:
                result = chain.invoke({"input_documents": docs, "question": query})
                answer = result["output_text"] if isinstance(result, dict) and "output_text" in result else chain.run(input_documents=docs, question=query)
            
            st.session_state.chat_history.append({
                "type": "bot",
                "content": answer,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            # Convert answer to speech
            with st.spinner("Preparing audio response..."):
                tts = gTTS(text=answer, lang='en')
                audio_path = os.path.join(tempfile.gettempdir(), "answer.mp3")
                tts.save(audio_path)
                
                with open(audio_path, "rb") as audio_file:
                    st.audio(audio_file.read(), format="audio/mp3")
                
                os.remove(audio_path)
                
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")

# Display chat history with black text
st.subheader("📜 Chat History")

if not st.session_state.chat_history:
    st.info("Your conversation will appear here")
else:
    for message in reversed(st.session_state.chat_history):
        if message['type'] == 'user':
            st.markdown(f"""
            <div class="message-container" style="justify-content: flex-end;">
                <div style="max-width: 80%;">
                    <div class="user-message">
                        {message['content']}
                    </div>
                    <div class="timestamp" style="text-align: right;">
                        {message['timestamp']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="message-container" style="justify-content: flex-start;">
                <div style="max-width: 80%;">
                    <div class="bot-message">
                        {message['content']}
                    </div>
                    <div class="timestamp">
                        {message['timestamp']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
# Popup Summary Modal
import streamlit.components.v1 as components

# Generate summary from chat history
def generate_summary(chat_history):
    if not chat_history:
        return "No conversation available to summarize."
    summary = ""
    for msg in chat_history:
        prefix = "User" if msg['type'] == 'user' else "Assistant"
        summary += f"{prefix} ({msg['timestamp']}): {msg['content']}\n"
    return summary.strip()

# Add summary button and modal container
with st.container():
    if st.button("📌 Summarize Chat Interaction", use_container_width=True):
        chat_summary = generate_summary(st.session_state.chat_history)
        st.session_state.show_summary_popup = True
        st.session_state.chat_summary = chat_summary

# Render popup modal if triggered
if st.session_state.get("show_summary_popup", False):
    components.html(f"""
    <div id="popup-summary" style="
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 350px;
        max-height: 300px;
        background-color: white;
        border: 2px solid #ccc;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        z-index: 9999;
        padding: 1rem;
        overflow-y: auto;
        font-family: sans-serif;
        color: black;
    ">
        <h4 style="margin-top: 0;">🧠 Chat Summary</h4>
        <pre style="white-space: pre-wrap; font-size: 13px;">{st.session_state.chat_summary}</pre>
        <button onclick="document.getElementById('popup-summary').style.display='none'"
            style="margin-top: 10px; background-color: #4f46e5; color: white; border: none; border-radius: 6px; padding: 6px 12px; cursor: pointer;">
            Close
        </button>
    </div>
    """, height=350)

# Footer
st.markdown("---")
st.caption(f"AI Teacher Assistant v1.0 · {datetime.now().strftime('%H:%M')} · Secure AI-powered learning")