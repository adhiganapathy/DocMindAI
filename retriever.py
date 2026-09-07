import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq

class DocumentRAG:
    def __init__(self, index_name: str = "document-ai-v2"):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = pc.Index(index_name)
        
        print("Initializing Groq AI Client...")
        self.client = Groq(api_key="")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_embedding = self.encoder.encode(query).tolist()
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        matches = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            matches.append({
                "score": match.get("score"),
                "source": metadata.get("source", "Unknown"),
                "doc_type": metadata.get("doc_type", "Unknown"),
                "text": metadata.get("text", "")
            })
            
        return matches

    def answer_query(self, query: str) -> dict:
        matches = self.search(query, top_k=5)
        if not matches:
            return {
                "answer": "I couldn't find any relevant information in your uploaded documents.",
                "matches": []
            }
            
        # Cleanly structure the context chunks for the LLM
        context_blocks = []
        for i, m in enumerate(matches, 1):
            context_blocks.append(f"--- Document Chunk {i} (Source: {m['source']}) ---\n{m['text']}")
        
        context_string = "\n\n".join(context_blocks)
        source_name = matches[0]["source"]
        
        system_prompt = (
            "You are an expert Document Intelligence Assistant. Your job is to analyze the provided document "
            "text chunks and give a direct, concise, and accurate answer to the user's question.\n"
            "- Ignore irrelevant boilerplate text, terms and conditions, or bank safety disclaimers.\n"
            "- Focus explicitly on line items, quantities, prices, invoice numbers, names, or requested data.\n"
            "- If the exact answer cannot be found in the text, state clearly what is available."
        )
        
        user_prompt = f"Document Context:\n{context_string}\n\nUser Question: {query}"
        
        try:
            completion = self.client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=512
            )
            raw_answer = completion.choices[0].message.content.strip()
            answer = f"Based on your document **{source_name}**:\n\n{raw_answer}"
        except Exception as e:
            answer = f"Error generating response from Groq: {str(e)}"
            
        return {
            "answer": answer,
            "matches": matches
        }