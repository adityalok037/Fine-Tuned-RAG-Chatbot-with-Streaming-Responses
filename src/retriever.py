import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorRetriever:
    def __init__(self, index_path: str = "../vectordb/faiss_index",
                 model_name: str = "all-MiniLM-L6-v2"):

        # Load FAISS index
        self.index = faiss.read_index(f"{index_path}/index.faiss")

        # Load corresponding text chunks
        with open(f"{index_path}/texts.pkl", "rb") as f:
            self.text_chunks = pickle.load(f)

        # Load the embedding model
        self.embedding_model = SentenceTransformer(model_name)

        print(f"Retriever initialized with {len(self.text_chunks)} chunks.")

    def retrieve(self, query: str, top_k: int = 5):
        # Embed the query
        query_vector = self.embedding_model.encode([query], convert_to_numpy=True)

        # Search for top_k similar chunks
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.text_chunks):
                results.append(self.text_chunks[idx])

        return results
