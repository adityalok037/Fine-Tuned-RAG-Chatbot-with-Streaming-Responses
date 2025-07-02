# src/pipeline.py

from src.retriever import VectorRetriever
from src.generator import GeminiFlashGenerator

class RAGPipeline:
    def __init__(self, index_path="../vectordb/faiss_index", top_k=4):
        """
        Initialize retriever and generator components.
        """
        self.retriever = VectorRetriever(index_path=index_path)
        self.generator = GeminiFlashGenerator()
        self.top_k = top_k

    def run(self, query: str, refine_query: bool = True) -> dict:
        """
        Full RAG pipeline:
        1. Optionally refines the user query
        2. Retrieves top-k relevant chunks
        3. Generates response using Gemini
        4. Returns both answer and source chunks
        """
        # Step 1: Refine query (optional)
        if refine_query:
            refined_query = self.generator.refine_query(query)
        else:
            refined_query = query

        # Step 2: Retrieve relevant docs
        retrieved_chunks = self.retriever.retrieve(refined_query, top_k=self.top_k)

        # Step 3: Generate answer using Gemini
        answer = self.generator.generate(refined_query, retrieved_chunks)

        return {
            "query": query,
            "refined_query": refined_query,
            "answer": answer,
            "source_chunks": retrieved_chunks
        }
