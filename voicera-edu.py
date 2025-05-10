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

# Load Cohere API key
cohere_api_key = st.secrets["cohere_api_key"]

# Custom CSS for professional look
st.markdown("""
<style>
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        padding: 20px;
        background-color: #f9fafb;
    }
    .chat-header {
        background-color: #2563eb;
        color: white;
        padding: 15px 20px;
        border-radius: 10px 10px 0 0;
        margin: -20px -20px 20px -20px;
    }
    .user-message {
        background-color: #e0e7ff;
        padding: 12px 16px;
        border-radius: 12px 12px 0 12px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
    }
    .bot-message {
        background-color: white;
        padding: 12px 16px;
        border-radius: 12px 12px 12px 0;
        margin: 8px 0;
        max-width: 80%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .status-message {
        background-color: #f3f4f6;
        padding: 10px 15px;
        border-radius: 8px;
        margin: 8px 0;
        text-align: center;
        font-size: 0.9em;
        color: #4b5563;
    }
    .file-uploader {
        border: 2px dashed #d1d5db;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
    .stAudio {
        width: 100%;
        margin-bottom: 20px;
    }
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 10px;
        object-fit: cover;
    }
    .message-container {
        display: flex;
        align-items: flex-start;
        margin-bottom: 15px;
    }
    .message-content {
        flex: 1;
    }
    .timestamp {
        font-size: 0.7em;
        color: #6b7280;
        margin-top: 4px;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="EduAI Assistant", layout="centered", page_icon="🧑‍🏫")

# Main container
with st.container():
    st.markdown("""
    <div class="chat-container">
        <div class="chat-header">
            <h2 style="margin:0; display:flex; align-items:center;">
                <img src="https://cdn-icons-png.flaticon.com/512/4712/4712035.png" class="avatar">
                EduAI Assistant
            </h2>
            <p style="margin:5px 0 0; font-size:0.9em; opacity:0.9;">Your personal AI teaching assistant</p>
        </div>
    """, unsafe_allow_html=True)

    # Display chat history
    for message in st.session_state.chat_history:
        if message['type'] == 'user':
            st.markdown(f"""
            <div class="message-container" style="justify-content:flex-end;">
                <div class="message-content">
                    <div class="user-message">{message['content']}</div>
                    <div class="timestamp">{message['timestamp']}</div>
                </div>
                <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" class="avatar">
            </div>
            """, unsafe_allow_html=True)
        elif message['type'] == 'bot':
            st.markdown(f"""
            <div class="message-container">
                <img src="https://cdn-icons-png.flaticon.com/512/4712/4712035.png" class="avatar">
                <div class="message-content">
                    <div class="bot-message">{message['content']}</div>
                    <div class="timestamp">{message['timestamp']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif message['type'] == 'status':
            st.markdown(f"""
            <div class="status-message">
                {message['content']}
            </div>
            """, unsafe_allow_html=True)

    # File uploader section
    st.markdown("""
    <div class="file-uploader">
        <h4 style="margin-top:0;">Upload your learning materials</h4>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
    doc_text = ""
    docsearch = None
    chain = None

    if uploaded_file:
        with st.spinner("Processing document..."):
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                if page.extract_text():
                    doc_text += page.extract_text().strip() + "\n"
            
            text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_text(doc_text)
            
            embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key, model="embed-english-v3.0")
            docsearch = FAISS.from_texts(texts, embeddings)
            
            llm = Cohere(cohere_api_key=cohere_api_key, temperature=0.3)
            chain = load_qa_chain(llm, chain_type="stuff")
            
            st.session_state.chat_history.append({
                'type': 'status',
                'content': f"📚 Document processed successfully with {len(texts)} sections",
                'timestamp': "Now"
            })
            st.rerun()

    # Input section
    st.markdown("""
    <div style="margin-top:20px; border-top:1px solid #e5e7eb; padding-top:20px;">
        <h4 style="margin-bottom:10px;">Ask your question</h4>
    </div>
    """, unsafe_allow_html=True)
    
    query = ""
    audio_bytes = st.audio_input("Speak your question", label_visibility="collapsed")

    if audio_bytes is not None:
        try:
            temp_dir = tempfile.mkdtemp()
            temp_audio_path = os.path.join(temp_dir, "input.webm")
            with open(temp_audio_path, "wb") as f:
                f.write(audio_bytes.getvalue())
            
            try:
                audio = AudioSegment.from_file(temp_audio_path)
                wav_path = os.path.join(temp_dir, "input.wav")
                audio.export(wav_path, format="wav")
                
                recognizer = sr.Recognizer()
                with sr.AudioFile(wav_path) as source:
                    audio_data = recognizer.record(source)
                query = recognizer.recognize_google(audio_data)
                
                st.session_state.chat_history.append({
                    'type': 'user',
                    'content': query,
                    'timestamp': "Now"
                })
                st.rerun()
            except Exception as e:
                st.error(f"Audio conversion failed: {e}")

            for file_path in [temp_audio_path, wav_path]:
                if os.path.exists(file_path):
                    os.remove(file_path)
            os.rmdir(temp_dir)
        except Exception as e:
            st.error(f"Voice recognition failed: {e}")

    query = st.text_input("Type your question here:", value=query, label_visibility="collapsed")

    # Generate answer when query is submitted
    if query and docsearch and chain:
        if not any(msg['content'] == query and msg['type'] == 'user' for msg in st.session_state.chat_history):
            st.session_state.chat_history.append({
                'type': 'user',
                'content': query,
                'timestamp': "Now"
            })
            st.rerun()
            
        with st.spinner("Generating response..."):
            docs = docsearch.similarity_search(query)

            if not docs:
                answer = "Sorry, I couldn't find relevant information in the document."
            else:
                try:
                    result = chain.invoke({"input_documents": docs, "question": query})
                    answer = result["output_text"] if isinstance(result, dict) and "output_text" in result else chain.run(input_documents=docs, question=query)
                except Exception as e:
                    answer = f"An error occurred while generating the answer: {str(e)}"

            st.session_state.chat_history.append({
                'type': 'bot',
                'content': answer,
                'timestamp': "Now"
            })
            
            # Text-to-Speech
            with st.spinner("Preparing audio response..."):
                tts = gTTS(text=answer, lang='en')
                audio_path = os.path.join(tempfile.gettempdir(), "answer.mp3")
                tts.save(audio_path)

                with open(audio_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format="audio/mp3", start_time=0)

                    # Auto-play the audio
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
            
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)  # Close chat-container

# Footer
st.markdown("""
<div style="text-align:center; margin-top:30px; color:#6b7280; font-size:0.8em;">
    <p>EduAI Assistant v1.0 · Secure AI-powered learning</p>
</div>
""", unsafe_allow_html=True)