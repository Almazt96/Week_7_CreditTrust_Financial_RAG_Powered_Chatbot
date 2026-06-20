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
