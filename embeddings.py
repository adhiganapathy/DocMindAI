import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

class DocumentEmbedder:
    def __init__(self, index_name: str = "document-ai-index"):
        print("Loading Embedding Model...")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index_name = index_name
        
        # Initialize Pinecone using the environment variable
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Create serverless index if it doesn't already exist
        existing_indexes = [idx["name"] for idx in pc.list_indexes()]
        if self.index_name not in existing_indexes:
            pc.create_index(
                name=self.index_name,
                dimension=384,  # matches all-MiniLM-L6-v2 vector size
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            
        self.index = pc.Index(self.index_name)

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunks.append(" ".join(words[i:i + chunk_size]))
        return chunks

    def upsert_document(self, doc_id: str, text: str, metadata: dict):
        chunks = self.chunk_text(text)
        vectors = []
        
        for idx, chunk in enumerate(chunks):
            embedding = self.encoder.encode(chunk).tolist()
            vector_id = f"{doc_id}-chunk-{idx}"
            chunk_metadata = {
                **metadata,
                "text": chunk,
                "chunk_index": idx
            }
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": chunk_metadata
            })
            
        self.index.upsert(vectors=vectors)
        print(f"Successfully embedded and upserted {len(vectors)} chunks for {doc_id} into Pinecone.")