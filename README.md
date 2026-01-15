# Voicera: Conversational AI for Education

**Voicera** is a production-grade AI assistant designed for educational environments. It combines automatic speech recognition (ASR), Retrieval-Augmented Generation (RAG), and text-to-speech (TTS) to facilitate multimodal interaction with academic content.

The platform operates in two configurations:
* **Syllabic Assistant:** Optimized for preloaded curricular content.
* **Universal Assistant:** Dynamic support for user-uploaded PDF documentation.

---

## Key Capabilities

* **Multimodal Interaction:** Supports seamless voice and text-based querying.
* **Automated RAG Pipeline:** Efficient PDF parsing, semantic chunking, and vectorization.
* **Context-Aware AI:** leverages Cohere for high-fidelity semantic search and response generation.
* **Audio Synthesis:** Integrated gTTS engine for audible response delivery.

---

## System Architecture

![Voicera System Architecture](https://github.com/user-attachments/assets/b24d634b-6672-452d-a22b-165b70c6000c)

---

## Technical Specifications

### Technology Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Frontend** | Streamlit | UI & Session Management |
| **Orchestration** | LangChain | Pipeline Management |
| **Vector Store** | FAISS | Similarity Search |
| **LLM / Embeddings** | Cohere | Generation & Embedding Models |
| **ASR** | SpeechRecognition | Voice-to-Text Processing |
| **TTS** | gTTS | Text-to-Speech Synthesis |

### Repository Structure

```text
voicera-ai/
├── voicera-edu.py          # Universal Assistant (PDF Upload)
├── voicera-ssc.py          # Syllabic Assistant (Preloaded)
├── requirements.txt        # Dependencies
├── .streamlit/
│   └── secrets.toml        # API Configuration
└── README.md               # Documentation
```
### Configuration
```
[general]
cohere_api_key = "your-cohere-api-key"
```
Replace "your-cohere-api-key" with your actual API key from Cohere.

Please not: The API is rate-limited. Large document sizes can exceed the rate limit of 10,0000 tokens per minute.

---
## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/voicera-ai.git
cd voicera-ai

# Install dependencies
pip install -r requirements.txt
