import sys, json
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

with open("data/golden_eval_set.json", "r", encoding="utf-8") as f:
    gold = json.load(f)

with open("data/eval_checkpoint.json", "r", encoding="utf-8") as f:
    results = json.load(f)

min_len = min(len(results[k]) for k in results)
gold_subset = gold[:min_len]

rag_res = results["Proposed System (RAG Agent)"][:min_len]
llm_res = results["Baseline 2 (Zero-shot LLM)"][:min_len]
triv_res = results["Baseline 1 (Trivial Canned)"][:min_len]

rag_wins = []
llm_wins = []
disagreements = []

for i in range(min_len):
    g = gold_subset[i]
    r = rag_res[i]
    l = llm_res[i]
    
    r_score = r.get("judge", {}).get("overall_score") or 0.0
    l_score = l.get("judge", {}).get("overall_score") or 0.0
    
    if r_score > l_score + 0.5:
        rag_wins.append({
            "idx": i+1,
            "id": g["id"],
            "query": g["customer_text"],
            "gold_intent": g["gold_intent"],
            "rag_score": r_score,
            "llm_score": l_score,
            "rag_reply": r["draft_reply"],
            "llm_reply": l["draft_reply"],
            "rag_reason": r.get("judge", {}).get("feedback", "")
        })
    elif l_score > r_score + 0.5:
        llm_wins.append({
            "idx": i+1,
            "id": g["id"],
            "query": g["customer_text"],
            "gold_intent": g["gold_intent"],
            "rag_score": r_score,
            "llm_score": l_score,
            "rag_reply": r["draft_reply"],
            "llm_reply": l["draft_reply"],
            "rag_feedback": r.get("judge", {}).get("feedback", ""),
            "llm_feedback": l.get("judge", {}).get("feedback", "")
        })

print(f"Total Evaluated: {min_len}")
print(f"RAG Won decisively (>0.5 score diff): {len(rag_wins)} cases")
print(f"Zero-shot LLM Won decisively (>0.5 score diff): {len(llm_wins)} cases")

print("\n--- SAMPLE 3 CASES WHERE RAG BEAT ZERO-SHOT ---")
for w in rag_wins[:3]:
    print(f"\n[Query #{w['idx']}] {w['query']}")
    print(f"RAG Score: {w['rag_score']} | LLM Score: {w['llm_score']}")
    print(f"RAG Reply: {w['rag_reply']}")
    print(f"LLM Reply: {w['llm_reply']}")

print("\n--- SAMPLE 3 CASES WHERE ZERO-SHOT BEAT RAG ---")
for w in llm_wins[:3]:
    print(f"\n[Query #{w['idx']}] {w['query']}")
    print(f"RAG Score: {w['rag_score']} | LLM Score: {w['llm_score']}")
    print(f"RAG Feedback: {w['rag_feedback']}")
    print(f"LLM Feedback: {w['llm_feedback']}")
    print(f"RAG Reply: {w['rag_reply']}")
    print(f"LLM Reply: {w['llm_reply']}")
