import os
import time
import streamlit as st
import chromadb

# Suppress HuggingFace Hub offline status warnings if executing local models
os.environ["HF_HUB_OFFLINE"] = "1"

from dotenv import load_dotenv
# CRITICAL: This must run BEFORE get_rag_system() is called
load_dotenv()

# Import your production pipeline class from your src folder
from src.rag_pipeline import CrediTrustRAG

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CrediTrust RAG Support Interface",
    page_icon="💬",
    layout="wide"
)

st.title("💬 CrediTrust RAG Support Interface")
st.caption("Empowering compliance, support, and product management teams with verified insight lineages.")

# --- 2. INITIALIZE PRODUCTION RAG BACKEND ---
@st.cache_resource
def get_rag_system():
    return CrediTrustRAG()

# Initialize the RAG variable
rag_system = get_rag_system()

# --- 3. SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. SIDEBAR ACTIONS ---
with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown(
        "**Compliance & Audit Mode**\n\n"
        "This portal provides strict, evidence-backed answers directly from the CFPB complaint logs. "
        "Every response forces complete transparency down to individual source snippets."
    )

# --- 5. RENDER CHAT HISTORY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("🔍 View Retrieved Sources & Evidence"):
                for idx, text_snippet in enumerate(msg["sources"], 1):
                    st.markdown(f"**Source Document {idx}:**")
                    st.info(f"*{text_snippet}*")

# --- 6. CHAT INPUT & EXECUTION ---
if user_input := st.chat_input("Ask a compliance or product support question..."):
    
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 2. Call Production RAG Pipeline Backend
    with st.spinner("Analyzing complaint archives and verifying sources..."):
        try:
            answer_text, context_chunks = rag_system.query(user_input)
        except Exception as e:
            answer_text = f"[ERROR] Execution failed: {str(e)}"
            context_chunks = []
            
    # 3. Create a Streamable Generator for the Answer
    def stream_response(text: str):
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.04) # Smooth visual playback streaming

    # 4. Display Assistant Message with Streaming
    with st.chat_message("assistant"):
        response_text = st.write_stream(stream_response(answer_text))
        
        # 5. Render Evidence Block if chunks were retrieved
        if context_chunks:
            with st.expander("🔍 View Retrieved Sources & Evidence", expanded=True):
                for idx, text_snippet in enumerate(context_chunks, 1):
                    st.markdown(f"**Source Document {idx}:**")
                    st.info(f"*{text_snippet}*")
                
    # 6. Save Assistant Response and Sources to Session State
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "sources": context_chunks
    })