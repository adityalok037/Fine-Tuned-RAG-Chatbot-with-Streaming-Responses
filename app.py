import os
import optparse
import streamlit as st
from src.pipeline import RAGPipeline


st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄📚 RAG Chatbot using Gemini 1.5 Flash")
st.markdown("Ask me anything based on your uploaded documents.")


@st.cache_resource
def load_rag_pipeline():
    return RAGPipeline(index_path="./vectordb/faiss_index", top_k=4)

rag = load_rag_pipeline()

with st.sidebar:
    st.markdown("### ℹ️ Model Info")
    st.write("LLM: `gemini-1.5-flash`")
    st.write(f"Chunks Indexed: {len(rag.retriever.text_chunks)}")
    st.markdown("---")
    st.button("🔄 Clear Chat", on_click=lambda: st.session_state.clear())

# Initialize session state for chat history
if "history" not in st.session_state:
    st.session_state["history"] = []

user_query = st.text_input("Enter your query:", placeholder="e.g. What is the refund policy?")

if st.button("Generate Response") and user_query:
    with st.spinner("Generating response..."):
        result = rag.run(user_query, refine_query=True)

        # Store chat history
        st.session_state.history.append({
            "query": user_query,
            "answer": result["answer"],
            "sources": result["source_chunks"]
        })

# Show Chat History
for chat in reversed(st.session_state["history"]):
    st.markdown(f"**🧑‍💼 You:** {chat['query']}")
    st.markdown(f"**🤖 Gemini:** {chat['answer']}")

    st.markdown("📚 **View source chunks used:**")
    for i, src in enumerate(chat["sources"], 1):
        with st.expander(f"📄 Chunk {i}"):
            st.markdown(src)
