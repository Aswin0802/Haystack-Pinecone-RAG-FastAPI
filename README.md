# 🚀 RAG System with Haystack, Pinecone, Qwen3.5 & FastAPI

A Retrieval-Augmented Generation (RAG) application built with **Haystack**, **Pinecone**, **Sentence Transformers**, **Qwen3.5-9B**, and **FastAPI**.

The application allows users to ask questions about PDF documents. Relevant document chunks are retrieved from a vector database and provided as context to an LLM, which generates a grounded answer.

---

## 🧠 Architecture

```text
                    ┌──────────────────────┐
                    │      User Query      │
                    └──────────┬───────────┘
                               │
                               ▼
                ┌───────────────────────────┐
                │ Sentence Transformers     │
                │      Text Embedder        │
                └─────────────┬─────────────┘
                              │
                              │ 768-D Vector
                              ▼
                    ┌──────────────────┐
                    │     Pinecone     │
                    │   Vector Store   │
                    └────────┬─────────┘
                             │
                             │ Top-K Documents
                             ▼
                 ┌────────────────────────┐
                 │   Chat Prompt Builder │
                 │                        │
                 │ Query + Retrieved      │
                 │ Context                │
                 └────────────┬───────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │    Qwen3.5-9B    │
                    │       LLM        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Generated Answer │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │      FastAPI     │
                    │      Backend     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Web Interface  │
                    └──────────────────┘
```

---

## ✨ Features

- 📄 PDF document ingestion
- ✂️ Document preprocessing and chunking
- 🔢 Sentence Transformer embeddings
- 🗄️ Pinecone vector database
- 🔍 Semantic similarity search
- 📚 Top-K relevant document retrieval
- 🧩 Context-aware prompt construction
- 🤖 Qwen3.5-9B LLM integration
- 🧠 Qwen non-thinking mode for direct responses
- 🚫 "I don't know" fallback when the context does not contain the answer
- ⚡ FastAPI REST API
- 🌐 HTML/Jinja2 frontend
- 🔐 Environment-based API key management
- 🛠️ Modern Python dependency management with `uv`

---

# 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Haystack | RAG pipeline orchestration |
| Sentence Transformers | Text embeddings |
| Pinecone | Vector database |
| Qwen3.5-9B | Large Language Model |
| Hugging Face API | LLM inference |
| FastAPI | Backend API |
| Jinja2 / HTML | Frontend |
| PyPDF | PDF processing |
| python-dotenv | Environment variable management |
| uv | Python package and environment management |

---

# 📂 Project Structure

```text
Haystack_Mistral_Pinecone_FastAPI/
│
├── app.py
├── pyproject.toml
├── uv.lock
├── .gitignore
├── README.md
│
├── templates/
│   └── index.html
│
├── Data/
│   └── RAG.pdf
│
└── QASystem/
    ├── __init__.py
    ├── retrieval_generation.py
    ├── utils.py
    └── ingest.py 
```

---

# 🔄 How the RAG System Works

## 1. Document Ingestion

The PDF is converted into Haystack documents, split into chunks, embedded, and stored in Pinecone.

```text
PDF
 ↓
PyPDFToDocument
 ↓
DocumentSplitter
 ↓
SentenceTransformerDocumentEmbedder
 ↓
Pinecone
```

---

## 2. Embedding Generation

Each document chunk is converted into a numerical vector.

```text
Document Chunk
      ↓
Sentence Transformer
      ↓
768-dimensional embedding
      ↓
Pinecone
```

The embedding dimension must match the Pinecone index dimension.

---

## 3. Query Processing

When a user asks a question:

```text
User Question
      ↓
Sentence Transformer
      ↓
Query Embedding
      ↓
Pinecone Similarity Search
```

Pinecone retrieves the most relevant document chunks.

---

## 4. Prompt Construction

The retrieved documents and user query are passed to a `ChatPromptBuilder`.

```text
Question
   +
Retrieved Context
   ↓
ChatPromptBuilder
   ↓
LLM Messages
```

The prompt instructs the model to use only the supplied context.

---

## 5. LLM Generation

The prompt is sent to **Qwen3.5-9B** through the Hugging Face API.

The application uses Qwen3.5 in non-thinking mode so the generator returns a direct answer rather than spending the available output on reasoning content.

```text
Retrieved Context
       ↓
Qwen3.5-9B
       ↓
Generated Answer
```

---

# 🧩 Haystack Query Pipeline

The query pipeline is:

```text
SentenceTransformersTextEmbedder
              │
              ▼
    PineconeEmbeddingRetriever
              │
              ▼
       ChatPromptBuilder
              │
              ▼
    HuggingFaceAPIChatGenerator
```

Important pipeline connections:

```python
query_pipeline.connect(
    "text_embedder.embedding",
    "retriever.query_embedding"
)

query_pipeline.connect(
    "retriever.documents",
    "prompt_builder.documents"
)

query_pipeline.connect(
    "prompt_builder.prompt",
    "llm.messages"
)
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_api_key
HF_TOKEN=your_huggingface_token
```
---

# ⚙️ Installation

## Prerequisites

Install:

- Python 3.13+
- Git
- `uv`
- A Pinecone account
- A Hugging Face account
- Pinecone API key
- Hugging Face API token

---

# 📥 Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/haystack-pinecone-rag-fastapi.git
```

```bash
cd haystack-pinecone-rag-fastapi
```

---

# 🐍 Setup with uv

Create the virtual environment:

```bash
uv venv myenv
```

Activate it on Windows Git Bash:

```bash
source myenv/Scripts/activate
```

Install dependencies:

```bash
uv sync
```

Or:

```bash
uv pip install -e .
```

---

# 🗄️ Pinecone Configuration

Create a Pinecone index with the configuration expected by this project:

```text
Index Name : haystack
Dimension  : 768
Metric     : cosine
Cloud      : AWS
Region     : us-east-1
Namespace  : rag
```

The vector dimension must match the embedding model used by the application.

---

# 📥 Document Ingestion

The ingestion pipeline processes the PDF and stores its vector representations in Pinecone.

The ingestion flow is:

```text
PDF
 ↓
Document Converter
 ↓
Document Splitter
 ↓
Document Embedder
 ↓
Pinecone Document Store
```

For large PDF sections, word-based splitting can be useful to prevent an individual chunk from becoming excessively large.

Example:

```python
DocumentSplitter(
    split_by="word",
    split_length=200,
    split_overlap=30
)
```

---

# ▶️ Run the Application

Start the FastAPI server:

```bash
python app.py
```

Or:

```bash
uvicorn app:app --reload
```

Open the application:

```text
http://127.0.0.1:8000
```

Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 🔌 API

## POST `/get_answer`

The endpoint accepts a user query and returns the generated answer.

Example:

```text
What is RAG?
```

Example JSON response:

```json
{
    "answer": "RAG stands for Retrieval-Augmented Generation..."
}
```

FastAPI handles JSON serialization automatically, so the endpoint can simply return:

```python
return {
    "answer": answer
}
```

---

# 🧪 Example RAG Flow

### User Query

```text
What is Retrieval-Augmented Generation?
```

### Processing

```text
1. Convert query into embedding
2. Search Pinecone
3. Retrieve top relevant chunks
4. Build chat prompt
5. Send prompt to Qwen3.5-9B
6. Generate answer
7. Return answer through FastAPI
```

### Example Output

```text
RAG stands for Retrieval-Augmented Generation. It combines
information retrieval with text generation by retrieving
relevant information and providing it as context to an LLM.
```

---

# 🛡️ Grounded Generation

The prompt instructs the model:

```text
Answer the following question using ONLY the provided context.

If the context does not contain an answer, reply with:
"I don't know"
```

This design helps reduce unsupported responses by restricting the model to retrieved information.

---

# 🧱 Key AI Engineering Concepts Demonstrated

## Retrieval

- Vector embeddings
- Semantic search
- Similarity search
- Top-K retrieval
- Vector databases

## RAG

- Document ingestion
- Document preprocessing
- Chunking
- Embedding generation
- Vector storage
- Context retrieval
- Prompt construction
- Grounded generation

## LLM Engineering

- Hugging Face inference APIs
- Qwen3.5-9B
- Chat-based generation
- Thinking vs non-thinking inference
- Generation parameters
- Context-aware prompting

## Backend Engineering

- FastAPI
- REST API development
- Request handling
- JSON responses
- Jinja2 templates

## Infrastructure

- Pinecone
- Environment variables
- API authentication
- `uv` dependency management

---

# 🚀 Future Improvements

The current implementation can be extended with:

- [ ] Hybrid search
- [ ] Metadata filtering
- [ ] Reranking models
- [ ] Query rewriting
- [ ] Query expansion
- [ ] Multi-query retrieval
- [ ] Conversation memory
- [ ] Streaming LLM responses
- [ ] Source citations
- [ ] Multi-document ingestion
- [ ] Advanced chunking strategies
- [ ] Retrieval evaluation
- [ ] RAGAS evaluation
- [ ] Observability and tracing
- [ ] Structured outputs
- [ ] Dockerization
- [ ] CI/CD
- [ ] Authentication and authorization
- [ ] Rate limiting
- [ ] Cloud deployment
- [ ] Production monitoring

---
---

# 🎯 Learning Objectives

This project demonstrates an end-to-end understanding of a modern RAG application:

```text
Documents
    ↓
Preprocessing
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Database
    ↓
Retrieval
    ↓
Prompt Construction
    ↓
LLM
    ↓
Grounded Response
    ↓
FastAPI
    ↓
User
```

The goal is to understand not only how to call an LLM, but how the individual components of a RAG system work together.

---

# 🔧 Troubleshooting

## Pinecone shows zero documents

Check:

```text
1. PINECONE_API_KEY
2. Index name
3. Namespace
4. Embedding dimension
5. Pinecone region
6. Whether ingestion completed successfully
```

---

## LLM returns an empty `.text`

Check the returned Haystack `ChatMessage`.

If the model is returning reasoning content without final text, verify that Qwen3.5 non-thinking mode is configured:

```python
generation_kwargs={
    "max_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.8,
    "extra_body": {
        "top_k": 20,
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }
}
```

Then retrieve the answer with:

```python
message = results["llm"]["replies"][0]

return message.text
```

---

## FastAPI cannot serialize `ChatMessage`

Do not return the complete Haystack `ChatMessage` object.

Use:

```python
return results["llm"]["replies"][0].text
```

Then in FastAPI:

```python
@app.post("/get_answer")
async def get_answer(query: str = Form(...)):
    answer = get_result(query)

    return {
        "answer": answer
    }
```

---



# 📌 Important Notes

- Pinecone index dimension must match the embedding dimension.
- The retriever uses the query embedding generated by the Sentence Transformer.
- The prompt builder receives the documents returned by Pinecone.
- `ChatPromptBuilder` is used because the application uses `HuggingFaceAPIChatGenerator`.
- The LLM receives the prompt through the `messages` input.
- Qwen3.5-9B supports thinking mode, but this project uses non-thinking mode for direct RAG responses.
- The application should never rely on an API key hardcoded in Python files.

---
# ⭐ Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.

