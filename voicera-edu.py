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
    page_title="AI Teacher Assistant",
    page_icon="🧑‍🏫",
    layout="centered"
)

# Custom CSS for professional styling
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
    
    .upload-card {
        border: 2px dashed var(--border);
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        background: var(--secondary);
        transition: all 0.3s;
    }
    
    .chat-message {
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        line-height: 1.5;
    }
    
    .user-message {
        background-color: #e0e7ff;
        border-radius: 12px 12px 0 12px;
        margin-left: auto;
        max-width: 80%;
    }
    
    .bot-message {
        background-color: white;
        border-radius: 12px 12px 12px 0;
        box-shadow: var(--card-shadow);
        max-width: 80%;
    }
    
    .timestamp {
        font-size: 0.7rem;
        color: var(--text-light);
        margin-top: 4px;
    }
    
    .summary-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.title("🎓 AI Teacher Chat Assistant")
st.markdown("Upload a textbook or syllabus (PDF), then ask a question by **voice** or **text** to get an instant spoken response.")

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
full_summary = ""

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
            
            # Create document summary
            full_summary = f"""
            ### 📚 Document Summary
            **File Name:** {uploaded_file.name}  
            **Size:** {uploaded_file.size/1024:.1f} KB  
            **Content Chunks:** {len(texts)}  
            
            #### Key Content Preview:
            {doc_text[:500]}{'...' if len(doc_text) > 500 else ''}
            """
            
            st.session_state.document_processed = True
            st.success(f"Document processed successfully ({len(texts)} sections)")
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")

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

# Display chat history
st.subheader("📜 Chat History")

if not st.session_state.chat_history:
    st.info("Your conversation will appear here")
else:
    for message in reversed(st.session_state.chat_history):
        if message['type'] == 'user':
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; margin-bottom: 8px;">
                <div style="max-width: 80%;">
                    <div class="chat-message user-message">
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
            <div style="display: flex; justify-content: flex-start; margin-bottom: 8px;">
                <div style="max-width: 80%;">
                    <div class="chat-message bot-message">
                        {message['content']}
                    </div>
                    <div class="timestamp">
                        {message['timestamp']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Document summary sidebar
with st.sidebar:
    st.header("Document Tools")
    
    if uploaded_file:
        with st.expander("📋 Document Summary", expanded=True):
            st.markdown(f"""
            <div class="summary-card">
                <h4 style="margin-top: 0;">{uploaded_file.name}</h4>
                <p><strong>Size:</strong> {uploaded_file.size/1024:.1f} KB</p>
                <p><strong>Sections:</strong> {len(texts) if 'texts' in locals() else 0}</p>
                <hr style="margin: 10px 0;">
                <h5>Content Preview:</h5>
                <div style="max-height: 200px; overflow-y: auto; padding: 8px; background: white; border-radius: 4px;">
                    {doc_text[:500]}{'...' if len(doc_text) > 500 else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("📥 Download Summary"):
                summary_text = f"Document: {uploaded_file.name}\n\nSummary:\n{doc_text[:2000]}"
                st.download_button(
                    label="Download",
                    data=summary_text,
                    file_name=f"{uploaded_file.name}_summary.txt",
                    mime="text/plain"
                )
    else:
        st.info("Upload a document to access summary tools")

# Footer
st.markdown("---")
st.caption("AI Teacher Assistant v1.0 · Secure AI-powered learning")