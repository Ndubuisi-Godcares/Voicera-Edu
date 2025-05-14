# Voicera 🎙️ – AI-Powered Conversational Assistant for Educational Content

**Voicera** is a production-grade conversational AI assistant designed for educational purposes. It enables voice and text-based interaction with syllabus content by combining speech recognition, LLM-powered document Q&A(Cohere), and voice synthesis into a seamless experience.

---

## 🧠 Key Features

- 🎤 **Multimodal Interaction**: Ask questions via voice or text.
- 📚 **PDF Parsing & Chunking**: Automatically extracts and processes academic documents.
- 🔍 **Contextual Q&A**: Answers user queries using Cohere’s LLM with context from your syllabus.
- 🔈 **Voice Output**: Converts AI responses into speech using Google Text-to-Speech (gTTS).
- 📄 **Conversation Summary**: Generates summaries of user interaction history.

---

## 🧩 System Architecture

```text

                            ┌─────────────────────┐
                            │   PDF Upload (UI)   │  ← User uploads educational PDF (e.g., textbook, syllabus)
                            └────────┬────────────┘
                                     ▼
                            ┌─────────────────────┐
                            │   PDF Text Extract  │  ← Extracts raw text content from the uploaded PDF using PyPDF2
                            └────────┬────────────┘
                                     ▼
                         ┌────────────────────────────┐
                         │  Text Chunking & Embedding │  ← Splits the extracted text into chunks & embeds using Cohere
                         │ (LangChain + Cohere)       │
                         └────────┬──────────────┬─────┘
                                  ▼              ▼
                       ┌────────────────┐  ┌────────────────────┐
                       │   FAISS Store  │  │  Cohere LLM (QA)   │  ← FAISS stores document embeddings; Cohere handles QA
                       └──────┬─────────┘  └──────────┬─────────┘
                              ▼                       ▼
                      ┌─────────────┐        ┌────────────────────┐
Voice/Text Input ────►│ User Query  │───────►│ Similarity Search   │  ← User submits query (text/voice); search for relevant docs
                      └────┬────────┘        └────────┬───────────┘
                           ▼                          ▼
                   ┌────────────────────┐     ┌────────────────────┐
                   │ Chat History State │◄────┤ LangChain QA Chain │  ← Chat history tracked; LangChain links QA processing
                   └────────┬───────────┘     └────────┬───────────┘
                            ▼                          ▼
            ┌──────────────────────────┐     ┌────────────────────┐
            │ Chat Display in Streamlit│     │ gTTS (Spoken Reply)│  ← Display chat history; convert text answer to speech
            └────────┬─────────────────┘     └────────┬───────────┘
                     ▼                                ▼
            ┌────────────────────┐          ┌────────────────────┐
            │ Chat Summary (UI) │          │ Audio Playback (UI)│  ← Summarize chat; Playback generated spoken response
            └────────────────────┘          └────────────────────┘


```

## 📁 Project Structure
```
voicera-ai/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── secrets.toml        # Secure API key storage
└── README.md               # Project documentation
```
### 🔐 Secrets Configuration
```
[general]
cohere_api_key = "your-cohere-api-key"
```
Replace "your-cohere-api-key" with your actual API key from Cohere.

---

## 🧰 Tech Stack

| Tool/Library            | Purpose                                      |
|-------------------------|----------------------------------------------|
| **Streamlit**           | UI & frontend logic                          |
| **LangChain**           | QA chain and text chunking                   |
| **FAISS**               | Vector store for similarity search           |
| **Cohere**              | LLM & embedding provider                     |
| **SpeechRecognition**   | Transcribe spoken questions                  |
| **gTTS**                | Convert responses to speech                  |
| **PyPDF2**              | PDF parsing and text extraction              |
| **pydub**               | Audio file conversion                        |

---

## 📝 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/voicera-ai.git
cd voicera-ai

# Install dependencies
pip install -r requirements.txt
