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
import streamlit.components.v1 as components
import base64
import uuid

# Load Cohere API key
cohere_api_key = st.secrets["cohere_api_key"]

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
    .audio-controls {
        width: 100%;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.title("🤖 Voicera - Conversational AI for Education")
st.caption("Upload a textbook or syllabus (PDF), then ask a question by voice or text to get an instant spoken response.")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
    
# Function to generate speech audio for a message
def generate_audio(text, message_id):
    try:
        tts = gTTS(text=text, lang='en')
        audio_path = os.path.join(tempfile.gettempdir(), f"response_{message_id}.mp3")
        tts.save(audio_path)
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        os.remove(audio_path)
        return audio_base64
    except Exception as e:
        st.error(f"Audio generation error: {str(e)}")
        return None

# Upload section (Collapsing section)
with st.expander("📄 Upload Learning Materials"):
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

# Document processing
if uploaded_file:
    with st.spinner("Processing document..."):
        try:
            doc_text = ""
            reader = PdfReader(uploaded_file)
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
            st.success(f"Document processed ({len(texts)} sections)")
        except Exception as e:
            st.error(f"Failed to process: {str(e)}")

# Sidebar document tools (Card layout)
with st.sidebar:
    st.header("📁 Document Tools")
    if uploaded_file:
        st.write(f"**Name:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size/1024:.1f} KB")
        st.write(f"**Sections:** {len(texts) if 'texts' in locals() else 0}")
        st.markdown("**Content Preview:**")
        st.markdown(f'<div class="document-content">{doc_text}</div>', unsafe_allow_html=True)
        st.download_button("📅 Download Text", doc_text, f"{uploaded_file.name}_content.txt")
    else:
        st.info("Upload a document to enable tools")

# Input (Collapsing section)
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
            st.session_state.chat_history.append({
                "type": "user", 
                "content": query, 
                "timestamp": datetime.now().strftime("%H:%M"),
                "id": str(uuid.uuid4())
            })
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
        st.session_state.chat_history.append({
            "type": "user", 
            "content": query, 
            "timestamp": datetime.now().strftime("%H:%M"),
            "id": str(uuid.uuid4())
        })
    with st.spinner("Answering your question..."):
        try:
            docs = docsearch.similarity_search(query)
            result = chain.invoke({"input_documents": docs, "question": query})
            answer = result.get("output_text", "I couldn't find a good answer.")
            
            # Generate a unique ID for the message
            msg_id = str(uuid.uuid4())
            
            # Generate audio for this response
            audio_base64 = generate_audio(answer, msg_id)
            
            # Add the bot's response to chat history with audio
            st.session_state.chat_history.append({
                "type": "bot", 
                "content": answer, 
                "timestamp": datetime.now().strftime("%H:%M"),
                "id": msg_id,
                "audio": audio_base64
            })
            
            # Auto-play the latest response
            st.markdown(f"""
                <audio id="latestResponseAudio" autoplay controls style="width: 100%;">
                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
                <script>
                    var audioElement = document.getElementById("latestResponseAudio");
                    audioElement.play();
                </script>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Response error: {str(e)}")

# Chat display
st.subheader("📜 Chat History")
if not st.session_state.chat_history:
    st.info("Your conversation will appear here.")
else:
    for msg in reversed(st.session_state.chat_history):
        bubble_class = "user-bubble" if msg['type'] == 'user' else "bot-bubble"
        audio_player = ""
        
        # Add audio player for bot messages if audio exists
        if msg['type'] == 'bot' and 'audio' in msg:
            audio_player = f"""
            <div class="audio-controls">
                <audio controls>
                    <source src="data:audio/mp3;base64,{msg['audio']}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
            </div>
            """
            
        st.markdown(f"""
        <div class='chat-bubble {bubble_class}'>
            {msg['content']}
            <div class='timestamp'>{msg['timestamp']}</div>
            {audio_player}
        </div>
        """, unsafe_allow_html=True)

# Chat summary
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