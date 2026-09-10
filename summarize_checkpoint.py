import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

with open("data/golden_eval_set.json", "r", encoding="utf-8") as f:
    gold = json.load(f)

with open("data/eval_checkpoint.json", "r", encoding="utf-8") as f:
    results = json.load(f)

min_len = min(len(results[k]) for k in results)
print(f"Total Evaluated Queries in Checkpoint: {min_len}")

gold_subset = gold[:min_len]

def compute_metrics(gold_items, system_results):
    gold_intents = [g["gold_intent"] for g in gold_items]
    pred_intents = [r["intent"] for r in system_results]
    intent_acc = accuracy_score(gold_intents, pred_intents)
    intent_f1 = f1_score(gold_intents, pred_intents, average="macro", zero_division=0)

    gold_esc = [1 if g["gold_escalate"] else 0 for g in gold_items]
    pred_esc = [1 if r["escalation_decision"] == "ESCALATE" else 0 for r in system_results]
    esc_prec = precision_score(gold_esc, pred_esc, zero_division=0)
    esc_rec = recall_score(gold_esc, pred_esc, zero_division=0)
    esc_f1 = f1_score(gold_esc, pred_esc, zero_division=0)

    judge_scores = [r["judge"]["overall_score"] for r in system_results if r.get("judge", {}).get("overall_score") is not None]
    avg_quality = np.mean(judge_scores) if judge_scores else 0.0

    groundedness_scores = [r["judge"]["groundedness_score"] for r in system_results if r.get("judge", {}).get("groundedness_score") is not None]
    avg_groundedness = np.mean(groundedness_scores) if groundedness_scores else 0.0

    return {
        "intent_acc": intent_acc,
        "intent_macro_f1": intent_f1,
        "esc_precision": esc_prec,
        "esc_recall": esc_rec,
        "esc_f1": esc_f1,
        "avg_quality_score": avg_quality,
        "avg_groundedness": avg_groundedness,
    }

summary_rows = []
for name, sys_results in results.items():
    m = compute_metrics(gold_subset, sys_results[:min_len])
    summary_rows.append({
        "System": name,
        "Intent Accuracy": f"{m['intent_acc']:.1%}",
        "Escalation F1": f"{m['esc_f1']:.3f}",
        "Groundedness (1-5)": f"{m['avg_groundedness']:.2f}",
        "Reply Quality (1-5)": f"{m['avg_quality_score']:.2f}",
    })

summary_df = pd.DataFrame(summary_rows)
print("\nSCORECARD ON EVALUATED QUERIES:")
print("=" * 80)
print(summary_df.to_string(index=False))
print("=" * 80)
summary_df.to_csv("data/evaluation_summary.csv", index=False)
