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

st.set_page_config(page_title="AI Teacher Assistant", layout="centered")
st.title("🎓 AI Teacher Chat Assistant")

st.markdown("""
Upload a textbook or syllabus (PDF), then ask a question by **voice** or **text** to get an instant spoken response.
""")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Step 1: Upload PDF
uploaded_file = st.file_uploader("📄 Upload a syllabus or textbook (PDF)", type=["pdf"])
doc_text, docsearch, chain = "", None, None

if uploaded_file:
    with st.spinner("Processing PDF..."):
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                doc_text += text.strip() + "\n"

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(doc_text)

        embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key, model="embed-english-v3.0")
        docsearch = FAISS.from_texts(texts, embeddings)

        llm = Cohere(cohere_api_key=cohere_api_key, temperature=0.3)
        chain = load_qa_chain(llm, chain_type="stuff")

        st.success(f"✅ PDF processed into {len(texts)} chunks.")

# Step 2: Voice or Text Input
query = ""
audio_bytes = st.audio_input("🎤 Ask a question using your voice:")

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
        st.success(f"🗣 You asked: {query}")

        os.remove(webm_path)
        os.remove(wav_path)
        os.rmdir(temp_dir)
    except Exception as e:
        st.error(f"Voice recognition failed: {e}")

query = st.text_input("💬 Or type your question here:", value=query)

# Step 3: Answering
if query and docsearch and chain:
    with st.spinner("Thinking like a teacher..."):
        docs = docsearch.similarity_search(query)

        if not docs:
            answer = "Sorry, I couldn't find relevant information in the document."
        else:
            try:
                answer = chain.invoke({"input_documents": docs, "question": query})
                if isinstance(answer, dict) and "output_text" in answer:
                    answer = answer["output_text"]
            except:
                answer = chain.run(input_documents=docs, question=query)

    # Save to chat history
    st.session_state.chat_history.append((query, answer))

    # Text-to-speech
    tts = gTTS(text=answer, lang='en')
    audio_path = os.path.join(tempfile.gettempdir(), "answer.mp3")
    tts.save(audio_path)
    audio_file = open(audio_path, "rb")
    audio_bytes = audio_file.read()
    audio_file.close()
    os.remove(audio_path)

# Display chat interface
st.markdown("---")
st.subheader("🧠 Chat History")

for i, (q, a) in enumerate(reversed(st.session_state.chat_history), 1):
    with st.container():
        st.markdown(f"**🧑 You:** {q}")
        st.markdown(f"**🤖 AI Teacher:** {a}")
        tts = gTTS(text=a, lang='en')
        audio_path = os.path.join(tempfile.gettempdir(), f"answer_{i}.mp3")
        tts.save(audio_path)
        with open(audio_path, "rb") as af:
            st.audio(af.read(), format="audio/mp3")
        os.remove(audio_path)

# Animated Avatar
st.markdown("""
<div style='text-align: center;'>
  <lottie-player src='https://assets2.lottiefiles.com/packages/lf20_mf2zpxsb.json' background='transparent' speed='1' style='width: 250px; height: 250px;' loop autoplay></lottie-player>
</div>
""", unsafe_allow_html=True)
st.markdown("<script src='https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js'></script>", unsafe_allow_html=True)

st.caption("Made with ❤️ for better learning experiences.")
