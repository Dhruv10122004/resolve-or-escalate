"""
Comprehensive Evaluation Harness for Hiver Take-Home Assignment.
Evaluates 3 systems on the Golden Evaluation Set:
1. Trivial Baseline (Majority Rule + Static Canned FAQ)
2. Simple Baseline (Zero-shot LLM without RAG Grounding)
3. Proposed System (SpotifySupportAgent with RAG Grounding)

Calculates:
- Intent Accuracy & Macro-F1
- Escalation Precision, Recall, and F1
- LLM-as-a-Judge Quality Scores (Groundedness, Helpfulness, Tone, Escalation)
- Outputs Markdown summary table and saves detailed JSON results.
"""

import argparse
import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from agent import SpotifySupportAgent
from baselines import SimpleLLMBaseline, TrivialBaseline
from judge import SupportJudge


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluation on the Golden Eval Set."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of examples to evaluate (e.g. 70 for fast sample, omit for all 198).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between API requests for rate-limit pacing (default: 1.0s).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Ignore existing checkpoints and restart evaluation from query 1.",
    )
    return parser.parse_args()


def compute_metrics(gold_items, system_results):
    """Calculates intent classification and escalation triage metrics."""
    # 1. Intent Metrics
    gold_intents = [g["gold_intent"] for g in gold_items]
    pred_intents = [r["intent"] for r in system_results]

    intent_acc = accuracy_score(gold_intents, pred_intents)
    intent_f1 = f1_score(gold_intents, pred_intents, average="macro", zero_division=0)

    # 2. Escalation Metrics (Binary 1=ESCALATE, 0=AUTO_HANDLE)
    gold_esc = [1 if g["gold_escalate"] else 0 for g in gold_items]
    pred_esc = [
        1 if r["escalation_decision"] == "ESCALATE" else 0
        for r in system_results
    ]

    esc_acc = accuracy_score(gold_esc, pred_esc)
    esc_prec = precision_score(gold_esc, pred_esc, zero_division=0)
    esc_rec = recall_score(gold_esc, pred_esc, zero_division=0)
    esc_f1 = f1_score(gold_esc, pred_esc, zero_division=0)

    # 3. Judge Quality Metrics (Filter out None/failed evaluations)
    judge_scores = [
        r["judge"]["overall_score"]
        for r in system_results
        if r.get("judge", {}).get("overall_score") is not None
    ]
    avg_quality = np.mean(judge_scores) if judge_scores else 0.0

    groundedness_scores = [
        r["judge"]["groundedness_score"]
        for r in system_results
        if r.get("judge", {}).get("groundedness_score") is not None
    ]
    avg_groundedness = (
        np.mean(groundedness_scores) if groundedness_scores else 0.0
    )

    return {
        "intent_acc": intent_acc,
        "intent_macro_f1": intent_f1,
        "esc_precision": esc_prec,
        "esc_recall": esc_rec,
        "esc_f1": esc_f1,
        "avg_quality_score": avg_quality,
        "avg_groundedness": avg_groundedness,
    }


def main():
    args = parse_args()

    # 1. Load Golden Benchmark
    eval_path = "data/golden_eval_set.json"
    checkpoint_path = "data/eval_checkpoint.json"
    if not os.path.exists(eval_path):
        raise FileNotFoundError(
            f"Could not find `{eval_path}`. Run build_golden_eval_set.py first!"
        )

    with open(eval_path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    if args.sample:
        golden_set = golden_set[: args.sample]
        print(
            f"\nRunning in SAMPLE mode on {len(golden_set)} benchmark items..."
        )
    else:
        print(f"\nRunning on FULL benchmark: {len(golden_set)} items...")

    # 2. Check for Checkpoint
    system_keys = [
        "Baseline 1 (Trivial Canned)",
        "Baseline 2 (Zero-shot LLM)",
        "Proposed System (RAG Agent)",
    ]
    results = {name: [] for name in system_keys}
    start_idx = 0

    if not args.reset and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                saved_results = json.load(f)
            min_len = min(len(saved_results.get(name, [])) for name in system_keys)
            if min_len > 0:
                for name in system_keys:
                    results[name] = saved_results[name][:min_len]
                start_idx = min_len
                print(f"\n[Checkpoint Found] Resuming evaluation from query {start_idx + 1}/{len(golden_set)}...")
        except Exception as e:
            print(f"[Warning] Could not load checkpoint: {e}. Starting fresh.")

    # 3. Initialize Systems
    print("\n--- Initializing Systems ---")
    trivial = TrivialBaseline()
    simple_llm = SimpleLLMBaseline()
    rag_agent = SpotifySupportAgent()
    judge = SupportJudge()

    systems = {
        "Baseline 1 (Trivial Canned)": trivial,
        "Baseline 2 (Zero-shot LLM)": simple_llm,
        "Proposed System (RAG Agent)": rag_agent,
    }

    # 4. Run Evaluation Loop
    total_items = len(golden_set)
    for idx in range(start_idx, total_items):
        item = golden_set[idx]
        item_num = idx + 1
        q_text = item["customer_text"]
        history = item.get("conversation_history", None)
        print(
            f"\n[{item_num}/{total_items}] Query: '{q_text[:70]}...'"
            if len(q_text) > 70
            else f"\n[{item_num}/{total_items}] Query: '{q_text}'"
        )

        for name, system in systems.items():
            print(f"   ▶ [{name}] Generating prediction...", end=" ", flush=True)
            pred = system.handle_message(q_text, conversation_history=history)
            print("Done! Grading with Groq Judge...", end=" ", flush=True)

            # Grade with Judge
            judge_res = judge.evaluate_reply(
                customer_text=q_text,
                agent_reply=pred.get("draft_reply", ""),
                agent_decision=pred.get("escalation_decision", "AUTO_HANDLE"),
                gold_intent=item["gold_intent"],
                gold_escalate=item["gold_escalate"],
                gold_key_points=item["gold_key_points"],
            )
            score = judge_res.get("overall_score")
            score_str = f"{score:.1f}/5.0" if score is not None else "Failed"
            print(f"Done (Score: {score_str})", flush=True)

            # Combine result
            record = {
                "id": item["id"],
                "customer_text": q_text,
                "intent": pred.get("intent", "GENERAL_AND_AMBIGUOUS"),
                "draft_reply": pred.get("draft_reply", ""),
                "escalation_decision": pred.get(
                    "escalation_decision", "AUTO_HANDLE"
                ),
                "escalation_reason": pred.get("escalation_reason", ""),
                "judge": judge_res,
            }
            results[name].append(record)

            # Pacing delay between LLM calls
            if name != "Baseline 1 (Trivial Canned)":
                time.sleep(args.delay)

        # Save checkpoint after every completed query
        try:
            os.makedirs("data", exist_ok=True)
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Warning] Failed to save checkpoint: {e}")

    # 5. Compute and Print Scorecard
    print("\n" + "=" * 80)
    print("FINAL EVALUATION BENCHMARK SCORECARD")
    print("=" * 80)

    summary_rows = []
    # Use only evaluated golden_set items
    evaluated_gold = golden_set[: len(results["Proposed System (RAG Agent)"])]
    for name, sys_results in results.items():
        m = compute_metrics(evaluated_gold, sys_results)
        summary_rows.append(
            {
                "System": name,
                "Intent Accuracy": f"{m['intent_acc']:.1%}",
                "Escalation F1": f"{m['esc_f1']:.3f}",
                "Groundedness (1-5)": f"{m['avg_groundedness']:.2f}",
                "Reply Quality (1-5)": f"{m['avg_quality_score']:.2f}",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    # 6. Save Results & Summary
    os.makedirs("data", exist_ok=True)
    summary_df.to_csv("data/evaluation_summary.csv", index=False)
    with open("data/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nResults successfully saved to:")
    print(" - `data/evaluation_summary.csv`")
    print(" - `data/evaluation_results.json`")
    print(" - `data/eval_checkpoint.json`\n")


if __name__ == "__main__":
    main()
