"""Single source of truth for prompts. Imported by train.py and evaluate.py so they never drift."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BANKING77_LABELS_FILE = ROOT / "data" / "synth" / "banking77_labels.json"

SYSTEM_UC1 = (
    "You are a customer support assistant for an online retail company. "
    "Reply to the customer's message helpfully and politely. When you need customer- or "
    "company-specific details you don't know (order number, website URL, phone number, etc.), "
    "write a placeholder in double curly braces, e.g. {{Order Number}} or {{Customer Support Phone Number}}."
)


def _banking_labels() -> list[str]:
    return json.loads(BANKING77_LABELS_FILE.read_text(encoding="utf-8"))


def system_uc2() -> str:
    labels = ", ".join(_banking_labels())
    return (
        "You are a support triage assistant for a digital bank. For the customer's message, "
        "choose exactly one intent label from this list and write a short, safe reply.\n"
        f"Allowed intent labels: {labels}\n"
        'Respond with ONLY a JSON object of the form {"intent": "<label>", "reply": "<text>"} '
        "and nothing else."
    )


def system_prompt(task: str) -> str:
    if task == "support_reply":
        return SYSTEM_UC1
    if task == "banking_triage":
        return system_uc2()
    raise ValueError(f"unknown task {task}")


def build_messages(task: str, user_text: str, fewshot: list[dict] | None = None) -> list[dict]:
    """Chat messages for one example. fewshot = list of {"input","target"} rows from train."""
    msgs = [{"role": "system", "content": system_prompt(task)}]
    for ex in fewshot or []:
        msgs.append({"role": "user", "content": ex["input"]})
        msgs.append({"role": "assistant", "content": ex["target"]})
    msgs.append({"role": "user", "content": user_text})
    return msgs


def to_prompt_completion(task: str, row: dict) -> dict:
    """TRL conversational prompt/completion format -> loss only on the completion."""
    return {
        "prompt": build_messages(task, row["input"]),
        "completion": [{"role": "assistant", "content": row["target"]}],
    }
