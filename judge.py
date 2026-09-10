"""
judge.py
--------
Implements an LLM-as-a-Judge evaluator using Groq (Llama-3.3-70B-Versatile).
Grades customer support replies on 4 dimensions:
1. Groundedness (1-5)
2. Helpfulness & Actionability (1-5)
3. Brand Tone & Empathy (1-5)
4. Escalation Appropriateness (1-5)
"""

import json
import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv(override=True)


class SupportJudge:

    def __init__(self, model_name="openai/gpt-oss-120b"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing! Please set it in your .env file."
            )
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

        self.system_instruction = """You are an expert, unbiased quality evaluator for Customer Support AI Agents.
Your job is to rigorously grade an AI agent's draft reply and escalation decision against a customer query and expected gold criteria.

### SCORING RUBRIC (Scale 1 to 5):
1. Groundedness (1-5): Is the reply factually accurate and grounded in real Spotify troubleshooting procedures? (1 = pure hallucination, 5 = perfectly accurate).
2. Helpfulness (1-5): Does the reply provide clear, actionable steps, diagnostic questions, or direct instructions? (1 = useless/unhelpful, 5 = highly actionable).
3. Brand Tone (1-5): Is the tone warm, concise (<240 chars), professional, empathetic, and representative of modern support? (1 = rude/robotic, 5 = ideal brand voice).
4. Escalation Appropriateness (1-5): Is the decision (AUTO_HANDLE vs ESCALATE) correct given whether the issue involves private account data, money/hacks, or basic troubleshooting? (1 = dangerously wrong, 5 = perfect decision).

### OUTPUT FORMAT:
You MUST return ONLY a valid JSON object with these exact keys:
{
  "groundedness_score": <1-5 integer>,
  "helpfulness_score": <1-5 integer>,
  "tone_score": <1-5 integer>,
  "escalation_score": <1-5 integer>,
  "overall_score": <1.0-5.0 float average>,
  "feedback": "<1-2 sentence concise justification>"
}
"""

    def evaluate_reply(
        self,
        customer_text: str,
        agent_reply: str,
        agent_decision: str,
        gold_intent: str,
        gold_escalate: bool,
        gold_key_points: list,
        max_retries: int = 5,
    ) -> dict:
        gold_action = "ESCALATE" if gold_escalate else "AUTO_HANDLE"
        key_points_str = (
            "\n".join([f"- {kp}" for kp in gold_key_points])
            if gold_key_points
            else "N/A"
        )

        prompt = f"""
[Customer Query]:
"{customer_text}"

[Gold Expected Intent]: {gold_intent}
[Gold Expected Decision]: {gold_action}
[Gold Resolution Key Points]:
{key_points_str}

[Agent Generated Reply]:
"{agent_reply}"

[Agent Generated Escalation Decision]: {agent_decision}

Grade this response according to the rubric and return strictly JSON:
"""

        for attempt in range(max_retries):
            try:
                chat_completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_instruction},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                data = json.loads(chat_completion.choices[0].message.content)
                if "overall_score" not in data:
                    data["overall_score"] = round(
                        (
                            data.get("groundedness_score", 3)
                            + data.get("helpfulness_score", 3)
                            + data.get("tone_score", 3)
                            + data.get("escalation_score", 3)
                        )
                        / 4.0,
                        2,
                    )
                return data
            except Exception as e:
                err_str = str(e)
                if (
                    "429" in err_str or "rate limit" in err_str.lower()
                ) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 12
                    print(
                        f"\n[Groq Rate Limit] Waiting {wait_time}s for token bucket replenishment..."
                    )
                    time.sleep(wait_time)
                else:
                    if attempt == max_retries - 1:
                        return {
                            "groundedness_score": None,
                            "helpfulness_score": None,
                            "tone_score": None,
                            "escalation_score": None,
                            "overall_score": None,
                            "feedback": f"[JUDGE_SCORING_FAILED] Could not evaluate reply due to error: {err_str}",
                            "error": True,
                        }
                    time.sleep(3)
