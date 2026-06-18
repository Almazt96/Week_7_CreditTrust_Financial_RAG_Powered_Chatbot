# Gradio/streamlit app for the LLM agent interface
import gradio as gr
import streamlit as st
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Enterprise RAG Support Interface",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Enterprise RAG Support Interface")
st.caption("Empowering non-technical stakeholders with verified compliance & product insights.")

# --- 2. MOCK RAG BACKEND PIPELINE ---
# Replace this function with your actual backend RAG pipeline invocation
def query_rag_pipeline(user_query: str):
    """
    Simulates fetching data from your backend RAG pipeline.
    Should return:
      - response_generator: a generator or iterable yielding text tokens
      - sources: a list of dictionaries containing metadata
    """
    # Simulated response tokens
    sample_response = (
        f"Based on our compliance logs, the issue regarding '{user_query}' seems to stem "
        "from recent updates in our billing synchronization cadence. Enterprise accounts "
        "experienced a 45-minute propagation delay on March 12th."
    )
    
    def response_generator():
        for word in sample_response.split(" "):
            yield word + " "
            time.sleep(0.08)  # Simulates streaming latency
            
    # Simulated retrieved source chunks
    mock_sources = [
        {
            "complaint_id": "CMP-2026-8891",
            "sub_issue": "Account Sync Delay",
            "snippet": "Customer reported that billing tier updates did not reflect immediately upon subscription upgrade, causing an erroneous overage alert."
        },
        {
            "complaint_id": "CMP-2026-0412",
            "sub_issue": "Interface Mismatch",
            "snippet": "Dashboard displayed 'Standard Tier' while the backend database correctly processed the 'Enterprise Upgrade'. Discrepancy resolved after manual cache flush."
        }
    ]
    
    return response_generator(), mock_sources

# --- 3. SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. SIDEBAR ACTIONS ---
with st.sidebar:
    st.header("Controls")
    # Clear Conversation Option to flush system state memory
    if st.button("🔄 Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown(
        "**Stakeholder View Mode**\n"
        "Provides verified lineage for Support, Compliance, and Product Management teams."
    )

# --- 5. RENDER CHAT HISTORY ---
# Display existing messages from session state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # If the message has sources attached, render them inside an expander
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("🔍 View Retrieved Sources & Evidence"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Source {idx}:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"🆔 **Complaint ID:** {src['complaint_id']}")
                    with col2:
                        st.caption(f"🏷️ **Sub-Issue Category:** {src['sub_issue']}")
                    st.info(f"*{src['snippet']}*")

# --- 6. CHAT INPUT & EXECUTION ---
if user_input := st.chat_input("Ask a compliance or product support question..."):
    
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 2. Call RAG Pipeline
    # (In production, pass your active vector store / LLM chain here)
    response_stream, retrieved_sources = query_rag_pipeline(user_input)
    
    # 3. Display Assistant Message with Streaming
    with st.chat_message("assistant"):
        # Stream the response text
        response_text = st.write_stream(response_stream)
        
        # Render the metadata container directly underneath the streamed answer
        with st.expander("🔍 View Retrieved Sources & Evidence", expanded=True):
            for idx, src in enumerate(retrieved_sources, 1):
                st.markdown(f"**Source {idx}:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"🆔 **Complaint ID:** {src['complaint_id']}")
                with col2:
                    st.caption(f"🏷️ **Sub-Issue Category:** {src['sub_issue']}")
                st.info(f"*{src['snippet']}*")
                
    # 4. Save Assistant Response and Sources to Session State
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "sources": retrieved_sources
    })

""" Key Highlights of This Code:
Response Streaming: Utilizing st.write_stream() alongside a python generator loop 
guarantees that stakeholders see tokens outputting live, mitigating perceived latency.

State Management: st.session_state.messages preserves the historical conversation 
context so it behaves like a true conversational application. The "Clear Conversation" 
button resets this state cleanly.

Auditability & Trust: Sources are organized into a strict hierarchy leveraging columns 
for metadata layout (complaint_id, sub_issue) and a styled st.info block for raw snippet 
readability. """