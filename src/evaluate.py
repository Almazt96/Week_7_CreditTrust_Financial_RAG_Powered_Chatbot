""" Task_3: Qualitative Evaluation Suite (evaluate.py)
To satisfy the Qualitative Evaluation requirement, this script runs a built-in test suite of 
highly precise analytical questions, prints out the context versus the answer, and provides a 
structured interface for you to apply the 1–5 quality rubric. """

import pandas as pd
import sys
import os

# Adds the current file's directory to the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import CrediTrustRAG

def run_evaluation_suite():
    # Initialize the RAG system
    rag = CrediTrustRAG(db_path="./production_chroma")
    
    # 5-10 Highly precise analytical evaluation questions
    eval_questions = [
        "What are the primary reasons customers claim their credit card dispute requests were denied?",
        "Are there recurring complaints regarding hidden fees in international money transfers?",
        "What structural problems do consumers report when attempting to unlock a frozen savings account?",
        "How do overdraft fee structures systematically impact checking account complaints mentioned?",
        "What specific documentation do banks fail to provide during a debt collection verification process?",
        "This question tests hallucinations: What did the CEO of CrediTrust say about the 2026 stock market crash?" 
        # (The final question tests if your prompt successfully forces "I do not have enough information.")
    ]
    
    eval_results = []
    
    print("\n" + "="*50)
    print("STARTING QUALITATIVE EVALUATION RUN")
    print("="*50 + "\n")
    
    for idx, q in enumerate(eval_questions, 1):
        print(f"Processing Evaluation Query {idx}/{len(eval_questions)}...")
        answer, context = rag.query(q)
        
        eval_results.append({
            "Question ID": idx,
            "Question": q,
            "Generated Answer": answer,
            "Retrieved Context Snippets": context
        })
        
        # Print results directly to terminal for live human scoring
        print(f"\n[Q{idx}] Question: {q}")
        print(f"Generated Answer:\n{answer}\n")
        print("-" * 60)
        
    # Save test results to a file for review
    df_eval = pd.DataFrame(eval_results)
    df_eval.to_csv("rag_evaluation_report.csv", index=False)
    print("\n[SUCCESS] Evaluation suite run complete. Results saved to 'rag_evaluation_report.csv'.")
    
    print("\n--- 1-5 Quality Scoring Rubric Reminder ---")
    print("Score your rows in the CSV based on:")
    print(" 5 - Perfect: Answers the question exactly using only context. Zero hallucination.")
    print(" 4 - Good: Factual, but misses a minor supporting detail from the context.")
    print(" 3 - Marginal: Answered, but leans dangerously close to outside assumptions.")
    print(" 2 - Poor: Handled poorly, contains minor hallucinations or ignores context constraints.")
    print(" 1 - Failure: Completely hallucinated, ignored instructions, or failed 'not enough info' rule.")

if __name__ == "__main__":
    run_evaluation_suite()