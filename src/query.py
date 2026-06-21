Task_2: # RAG query Streamlined Interactive Interface
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag_pipeline import CrediTrustRAG

def main():
    print("--- Connecting to Central CrediTrust Framework ---")
    try:
        rag = CrediTrustRAG()
    except Exception:
        print("[CRITICAL] Could not locate or read the indexed vector store. Run indexer.py first!")
        return

    print("\nWelcome to the CFPB Complaint RAG Assistant!")
    print("Type your questions below to query indexed complaints. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_query = input("User Question: ").strip()
            if not user_query:
                continue
            if user_query.lower() in ['exit', 'quit']:
                print("Shutting down assistant. Goodbye!")
                break

            print(f"\n{'='*30} GENERATED ANALYSIS {'='*30}")
            answer, contexts = rag.query(user_query)
            print(f"\nResponse:\n{answer}\n")
            print('='*79 + '\n')

        except (KeyboardInterrupt, EOFError):
            print("\nShutting down assistant. Goodbye!")
            break

if __name__ == "__main__":
    main()