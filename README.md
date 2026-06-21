# Week_7_CreditTrust_Financial_RAG_Powered_Chatbot

Current Codebase Architecture & Git Compliance
To ensure industrial tracking across the 10 Academy cohorts, the repository structure complies exactly with standard continuous integration conventions. The prototype tasks have been successfully isolated into separate modular scripts:
rag-complaint-chatbot/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml       # Automates GitHub Actions unit tests
├── data/
│   ├── raw/                    # Store raw CFPB data here
│   └── processed/              # Stores filtered_complaints.csv
├── vector_store/               # Holds persisted FAISS/ChromaDB indices
├── notebooks/
│   ├── __init__.py
│   └── EDA_and_Preprocessing.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # Modular text cleaning logic
│   ├── indexer.py               # Chunking & embedding scripts
│   └── rag_pipeline.py          # Core retrieval & generator wrapper
├── tests/
│   ├── __init__.py
│   └── test_rag.py              # PyTest test cases
├── app.py                      # Core Gradio/Streamlit entry point application
├── requirements.txt            # Python production dependencies
├── README.md                   # Repository documentation and startup commands
└── .gitignore                  # Prevents committing datasets and local indexes

<!-- app.py -->
# Task_4: Gradio/streamlit app for the LLM agent interface
""" Task 4: Creating an Interactive Chat Interface
To empower non-technical support, compliance, and product managers, we will encapsulate our 
backend RAG pipeline into an interactive web UI.
Step-by-Step Implementation Protocol:
•	UI Framework Selection: Build the application interface in `app.py` utilizing Streamlit or 
Gradio. Streamlit is highly recommended for multi-column analytic dashboard layouts,
while Gradio fits perfectly into conversational interfaces.
•	Conversational & Input Components: Provide a clean, full-width text input box or a dedicated 
chat interface (`st.chat_input` or `gr.ChatInterface`). Implement a responsive submit button and 
a separate 'Clear Conversation' option to flush system state memory.
•	Source & Evidence Display: Crucial for enterprise trust! Design an expanding or dedicated 
secondary metadata container below or beside the AI answer. Loop over the retrieved source 
chunks, displaying the raw text narrative snippet, the actual `complaint_id`, and its corresponding sub-issue category.
•	Response Streaming (Recommended): Configure token-by-token generation streaming using g
enerator loops. This provides visual feedback and eliminates latency frustration for business stakeholders.
 """
""" Key Highlights of This Code:
Response Streaming: Utilizing st.write_stream() alongside a python generator loop 
guarantees that stakeholders see tokens outputting live, mitigating perceived latency.

State Management: st.session_state.messages preserves the historical conversation 
context so it behaves like a true conversational application. The "Clear Conversation" 
button resets this state cleanly.

Auditability & Trust: Sources are organized into a strict hierarchy leveraging columns 
for metadata layout (complaint_id, sub_issue) and a styled st.info block for raw snippet 
readability. """
