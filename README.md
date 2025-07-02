# RAG Chatbot using Gemini 1.5 Flash
An intelligent chatbot that answers user queries based on a custom document set (like Terms & Conditions, Policies, Legal Docs). It uses a Retrieval-Augmented Generation (RAG) pipeline and Google's Gemini 1.5 Flash model via API for real-time, grounded, and factual responses.

This is a complete end-to-end GenAI system: from document ingestion - vector store - semantic search - LLM-based generation - live chatbot via Streamlit.


# Module  Description

Document Ingestion --  Accepts and processes long PDF documents 
Chunking -- Splits documents into small, semantic-aware chunks 
Embeddings -- Uses `all-MiniLM-L6-v2` from SentenceTransformers 
Vector Store -- Stores chunks in FAISS DB for fast semantic search 
Generator --  Uses `gemini-1.5-flash` API to generate grounded answers 
Streamlit UI -- A modern chatbot interface with real-time response 
Source Reference -- Shows exactly which parts of the doc were used 
Reset/Clear Chat -- Built-in memory and state reset 


## 💬 How the RAG Chatbot Works (Deep Dive)

This chatbot follows the RAG (Retrieval-Augmented Generation) architecture using Gemini 1.5 Flash as the language model. Below are the internal components and their responsibilities:

---

# 1. Document Ingestion & Preprocessing

- The system reads one or more PDF files from the `/data` folder using `PyPDFLoader`.
- It cleans raw text (removing headers, footers, extra spaces).
- The text is split into manageable semantic chunks using LangChain’s `RecursiveCharacterTextSplitter`:
  - `chunk_size=500` words
  - `chunk_overlap=20` words (to preserve context)

# 2. Embedding Generation

- Each chunk is converted into a dense vector using `sentence-transformers/all-MiniLM-L6-v2`.
- Embeddings capture the semantic meaning of the text, which enables better matching with queries.


# 3. Vector Database (FAISS)

- All text embeddings are stored in FAISS — a local, fast vector similarity search engine.
- Each entry stores:
  - Embedding vector
  - Original text chunk
  - Metadata (e.g., source document ID)

# 4. Query Processing & Semantic Search

- When a user enters a query, it is also embedded using the same embedding model.
- FAISS performs a **k-nearest neighbor search** to find the most relevant document chunks.
- These chunks are passed as context for the language model to generate a factual, grounded answer.


# 5. Gemini Flash as the Generator

- The retrieved chunks + the user query are injected into a prompt template.
- Gemini 1.5 Flash is called via API using `google-generativeai` SDK.
- It returns a response that is:
  - Aligned with the document
  - More fluent and coherent than retrieval alone

# 6. Streamlit Interface

- Streamlit is used for building an interactive frontend:
  - Input text box for queries
  - Chat-like response display
  - Expandable view for seeing which chunks were used
  - Sidebar showing model name, number of chunks, and clear/reset button


### ✅ Output Flow Example
-- Output screenshots are attached in the screenshot folder

