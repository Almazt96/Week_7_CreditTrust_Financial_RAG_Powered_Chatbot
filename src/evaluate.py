""" Task_3: Qualitative Evaluation Suite (evaluate.py)
To satisfy the Qualitative Evaluation requirement, this script runs a built-in test suite of 
highly precise analytical questions, prints out the context versus the answer, and provides a 
structured interface for you to apply the 1–5 quality rubric. """

import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag_pipeline import CrediTrustRAG

def run_evaluation_suite():
    # Instantiate central pipeline engine
    rag = CrediTrustRAG()
    
    eval_questions = [
        "What are the primary reasons customers claim their credit card dispute requests were denied?",
        "Are there recurring complaints regarding hidden fees in international money transfers?",
        "What structural problems do consumers report when attempting to unlock a frozen savings account?",
        "How do overdraft fee structures systematically impact checking account complaints mentioned?",
        "What specific documentation do banks fail to provide during a debt collection verification process?",
        "This question tests hallucinations: What did the CEO of CrediTrust say about the 2026 stock market crash?"
    ]
    
    eval_results = []
    print("\n" + "="*50 + "\nSTARTING QUALITATIVE EVALUATION RUN\n" + "="*50)
    
    for idx, q in enumerate(eval_questions, 1):
        print(f"Processing Query {idx}/{len(eval_questions)}...")
        answer, context = rag.query(q, k=3)
        
        eval_results.append({
            "Question ID": idx,
            "Question": q,
            "Generated Answer": answer,
            "Retrieved Context Snippets": "||".join(context)
        })
        
        print(f"\n[Q{idx}] Question: {q}")
        print(f"Generated Answer:\n{answer}\n" + "-"*60)
        
    df_eval = pd.DataFrame(eval_results)
    df_eval.to_csv("data/processed/rag_evaluation_report.csv", index=False)
    print("\n[SUCCESS] Qualitative Evaluation Report written to: 'data/processed/rag_evaluation_report.csv'")

if __name__ == "__main__":
    run_evaluation_suite()