"""
Implements two comparative baseline systems required by the assignment:
1. TrivialBaseline: Majority intent + static canned FAQ template (Zero AI).
2. SimpleLLMBaseline: Zero-shot LLM without RAG knowledge base retrieval.
"""

import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class TrivialBaseline:
    """A naive rule-based baseline that returns majority class and static canned text."""

    def __init__(self, majority_intent="BILLING_AND_PAYMENTS"):
        self.majority_intent = majority_intent

    def handle_message(
        self, customer_text: str, conversation_history: list = None
    ) -> dict:
        return {
            "intent": self.majority_intent,
            "draft_reply": "Thanks for reaching out to Spotify! For help with your account or music, please visit https://support.spotify.com.",
            "escalation_decision": "AUTO_HANDLE",
            "escalation_reason": "Static default rule-based policy.",
        }


class SimpleLLMBaseline:
    """A naive zero-shot LLM baseline that answers without any RAG grounding context."""

    def __init__(self, model_name="gemini-3.6-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing! Please set it in your .env file."
            )
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

        self.system_instruction = """
You are a customer support agent for Spotify (@SpotifyCares).
Classify the incoming customer tweet, draft a reply, and decide whether to escalate.

### INTENTS:
- PLAYBACK_AND_TECHNICAL
- BILLING_AND_PAYMENTS
- ACCOUNT_AND_LOGIN
- CONTENT_AND_CATALOG
- FEATURE_AND_FEEDBACK
- GENERAL_AND_AMBIGUOUS

### OUTPUT FORMAT:
Return ONLY a valid JSON object:
{
  "intent": "<INTENT>",
  "draft_reply": "<Reply in Spotify voice>",
  "escalation_decision": "AUTO_HANDLE" or "ESCALATE",
  "escalation_reason": "<1-sentence reason>"
}
"""

    def handle_message(
        self, customer_text: str, conversation_history: list = None
    ) -> dict:
        prompt = f"Customer Tweet: \"{customer_text}\"\nProvide your JSON classification and reply:"

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            return json.loads(response.text)
        except Exception:
            return {
                "intent": "GENERAL_AND_AMBIGUOUS",
                "draft_reply": "Hi! How can we assist you with Spotify today?",
                "escalation_decision": "AUTO_HANDLE",
                "escalation_reason": "Fallback response.",
            }
