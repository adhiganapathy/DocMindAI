# DocMind AI

A production-ready, full-stack Document AI application designed to ingest unstructured documents, extract complex multi-item line data using custom OCR, and enable high-speed semantic search and conversational reasoning via vector embeddings and LLMs.

## Tech Stack

- **Frontend**: Streamlit (Containerized UI for document upload and interactive chat)
- **Backend**: FastAPI (High-performance asynchronous REST API)
- **OCR Engine**: PaddleOCR & OpenCV (Text extraction and layout parsing)
- **Vector Database**: Serverless Pinecone (Semantic chunk indexing)
- **LLM / Inference**: Groq API (`openai/gpt-oss-20b`) for ultra-low latency reasoning
- **Deployment**: Docker & Docker Compose (Multi-container architecture)

## Project Structure

```text
docmind-ai/
│
├── api/
│   └── main.py
├── classification/
│   └── classifier.py
├── extraction/
│   ├── layoutlm_extractor.py
│   └── ai_extractor.py
├── ocr/
│   ├── paddle_ocr.py
│   └── preprocessing.py
├── rag/
│   ├── embeddings.py
│   └── retriever.py
├── validation/
│   └── guardrails.py
├── app.py
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
└── .env

Clone the repository:
git clone [https://github.com/your-username/docmind-ai.git](https://github.com/your-username/docmind-ai.git)
cd docmind-ai

Create a .env file in the root directory:
PINECONE_API_KEY=your_pinecone_api_key_here
GROQ_API_KEY=your_groq_api_key_here


Build and run the entire stack:
docker compose up --build


Access the application:

Frontend UI: http://localhost:8501

Backend API Docs: http://localhost:8000/docs

