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
def process_document(file_bytes, file_name):
    """Process PDF document and return text chunks and vector store"""
    try:
        reader = PdfReader(BytesIO(file_bytes))
        doc_text = ""
        max_pages = 20
        
        for i, page in enumerate(reader.pages):
            if i >= max_pages:
                break
            text = page.extract_text()
            if text:
                doc_text += text.strip() + "\n"

        if not doc_text.strip():
            raise ValueError("No text could be extracted from the PDF")

        splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_text(doc_text)

        if not texts:
            raise ValueError("No text chunks created from the document")

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=st.secrets["gemini_api_key"]
        )
        docsearch = FAISS.from_texts(texts, embeddings)

        return doc_text, texts, docsearch
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return None, None, None

def cleanup_temp_files(*paths):
    """Safely cleanup temporary files and directories"""
    for path in paths:
        try:
            if os.path.exists(path):
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    os.rmdir(path)
        except Exception as e:
            st.warning(f"Could not cleanup {path}: {str(e)}")

# App Header
st.title("🤖 Voicera - Conversational AI for Education")
st.caption("Upload a textbook or syllabus (PDF), then ask a question by voice or text to get an instant spoken response.")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "audio_responses" not in st.session_state:
    st.session_state.audio_responses = {}
if "docsearch" not in st.session_state:
    st.session_state.docsearch = None
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""
if "texts" not in st.session_state:
    st.session_state.texts = []
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None

# Upload section
with st.expander("📄 Upload Learning Materials"):
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

# Process uploaded file
if uploaded_file:
    # Check if this is a new file
    if st.session_state.current_file_name != uploaded_file.name:
        file_bytes = uploaded_file.read()
        with st.spinner("Processing document..."):
            doc_text, texts, docsearch = process_document(file_bytes, uploaded_file.name)
            
            if doc_text and texts and docsearch:
                # Store in session state
                st.session_state.doc_text = doc_text
                st.session_state.texts = texts
                st.session_state.docsearch = docsearch
                st.session_state.document_processed = True
                st.session_state.current_file_name = uploaded_file.name
                st.success(f"Document processed successfully! ({len(texts)} sections)")
            else:
                st.error("Failed to process the document. Please try again.")
                st.session_state.document_processed = False
    else:
        # File already processed
        st.info(f"Document '{uploaded_file.name}' is already loaded.")
else:
    # No file uploaded - reset state if needed
    if st.session_state.current_file_name:
        st.session_state.document_processed = False
        st.session_state.docsearch = None
        st.session_state.doc_text = ""
        st.session_state.texts = []
        st.session_state.current_file_name = None

# Sidebar tools
with st.sidebar:
    st.header("📁 Document Tools")
    if st.session_state.document_processed and uploaded_file:
        st.write(f"**Name:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
        st.write(f"**Sections:** {len(st.session_state.texts)}")
        st.markdown("**Content Preview:**")
        st.markdown(f'<div class="document-content">{st.session_state.doc_text[:1000]}{"..." if len(st.session_state.doc_text) > 1000 else ""}</div>', unsafe_allow_html=True)
        st.download_button("📥 Download Text", st.session_state.doc_text, f"{uploaded_file.name}_content.txt")
        
        if st.button("🗑️ Clear Document"):
            st.session_state.document_processed = False
            st.session_state.docsearch = None
            st.session_state.doc_text = ""
            st.session_state.texts = []
            st.session_state.current_file_name = None
            st.success("Document cleared!")
            st.rerun()
    else:
        st.info("Upload a document to enable tools")

# Question input section
with st.expander("💬 Ask Your Question", expanded=st.session_state.document_processed):
    if not st.session_state.document_processed:
        st.warning("Please upload and process a document first!")
    else:
        query = ""
        
        # Voice input
        audio_bytes = st.audio_input("🎤 Speak your question:")
        if audio_bytes:
            temp_dir = None
            webm_path = None
            wav_path = None
            try:
                temp_dir = tempfile.mkdtemp()
                webm_path = os.path.join(temp_dir, "input.webm")
                wav_path = os.path.join(temp_dir, "input.wav")
                
                # Save audio file
                with open(webm_path, "wb") as f:
                    f.write(audio_bytes.getvalue())
                
                # Convert to WAV
                audio = AudioSegment.from_file(webm_path)
                audio.export(wav_path, format="wav")
                
                # Speech recognition
                recognizer = sr.Recognizer()
                with sr.AudioFile(wav_path) as source:
                    audio_data = recognizer.record(source)
                query = recognizer.recognize_google(audio_data)
                
                if query:
                    st.success(f"Recognized: {query}")
                
            except Exception as e:
                st.error(f"Speech recognition failed: {str(e)}")
            finally:
                cleanup_temp_files(webm_path, wav_path, temp_dir)

        # Text input
        text_query = st.text_input("💬 Or type your question:", value=query if query else "")
        final_query = text_query if text_query else query

# Process and answer the query
if final_query and st.session_state.document_processed and st.session_state.docsearch:
    # Check if this is a new query
    last_user_msg = None
    if st.session_state.chat_history:
        for msg in reversed(st.session_state.chat_history):
            if msg['type'] == 'user':
                last_user_msg = msg['content']
                break
    
    if final_query != last_user_msg:
        # Add user message to chat
        st.session_state.chat_history.append({
            "type": "user", 
            "content": final_query,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        
        with st.spinner("🤔 Analyzing your question..."):
            temp_dir = None
            audio_path = None
            try:
                # Get answer from document
                llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
                chain = load_qa_chain(llm, chain_type="stuff")
                docs = st.session_state.docsearch.similarity_search(final_query, k=3)
                result = chain.invoke({"input_documents": docs, "question": final_query})
                answer = result.get("output_text", "I couldn't find a good answer in the document.")
                
                # Add bot response to chat
                st.session_state.chat_history.append({
                    "type": "bot", 
                    "content": answer,
                    "timestamp": datetime.now().strftime("%H:%M")
                })

                # Generate TTS audio
                response_id = str(uuid.uuid4())
                tts = gTTS(text=answer, lang='en')
                temp_dir = tempfile.mkdtemp()
                audio_path = os.path.join(temp_dir, f"response_{response_id}.mp3")
                tts.save(audio_path)

                # Store audio in session state
                with open(audio_path, "rb") as audio_file:
                    st.session_state.audio_responses[response_id] = audio_file.read()

                # Play audio
                st.audio(st.session_state.audio_responses[response_id], format="audio/mp3")
                st.rerun()

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.session_state.chat_history.append({
                    "type": "bot", 
                    "content": f"Sorry, I encountered an error: {str(e)}",
                    "timestamp": datetime.now().strftime("%H:%M")
                })
            finally:
                cleanup_temp_files(audio_path, temp_dir)

# Chat History Display
st.subheader("💬 Chat History")
if not st.session_state.chat_history:
    st.info("Your conversation will appear here after you ask a question.")
else:
    # Display messages in chronological order (most recent at bottom)
    for msg in st.session_state.chat_history:
        bubble_class = "user-bubble" if msg['type'] == 'user' else "bot-bubble"
        icon = "👤" if msg['type'] == 'user' else "🤖"
        st.markdown(f"""
        <div class='chat-bubble {bubble_class}'>
            {icon} {msg['content']}
            <div class='timestamp'>{msg['timestamp']}</div>
        </div>
        """, unsafe_allow_html=True)

# Chat management buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("📋 Generate Chat Summary"):
        if st.session_state.chat_history:
            summary_text = "\n".join([
                f"{'User' if h['type'] == 'user' else 'AI'} ({h['timestamp']}): {h['content']}" 
                for h in st.session_state.chat_history
            ])
            st.text_area("📋 Chat Summary:", summary_text, height=200)
        else:
            st.info("No conversation to summarize yet.")

with col2:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.audio_responses = {}
        st.success("Chat history cleared!")
        st.rerun()

# Footer
st.markdown("---")
st.markdown("*Voicera uses Google's Gemini AI and text-to-speech to provide interactive learning experiences.*")