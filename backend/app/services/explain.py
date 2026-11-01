import json
import time

from fastapi import HTTPException
from pydantic import BaseModel, TypeAdapter, ValidationError

from .. import settings

RETRIES = 4

INSTRUCTIONS = (
    "You are an architectural compliance assistant. Each item below is a check that a deterministic program has "
    "already measured and marked as failed. Do not re-measure, change a status, or quote clause numbers that are "
    "not given. For every item write one plain-language explanation of why it matters (one or two sentences) and "
    "one practical fix (one sentence). Return JSON only: an object whose keys are the item keys and whose values "
    'are {"explanation": string, "fix": string}.'
)


class Explanation(BaseModel):
    explanation: str
    fix: str


def prompt_for(violations):
    items = [{"key": v["key"], "rule": v["rule"], "room": v["room"], "severity": v["severity"],
              "measured": v["measured"], "required": v["required"], "margin_pct": v["margin_pct"],
              "clause": v["clause"]} for v in violations]
    return f"{INSTRUCTIONS}\n\n{json.dumps(items, indent=1)}"


def call_model(prompt):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    for attempt in range(RETRIES):
        try:
            response = client.models.generate_content(
                model=settings.GEMINI_MODEL, contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"))
            return response.text
        except Exception as e:
            if attempt == RETRIES - 1:
                raise HTTPException(502, f"explanation service failed: {type(e).__name__}")
            time.sleep(2 ** attempt)


def explain(report):
    if not settings.GEMINI_API_KEY:
        raise HTTPException(503, "GEMINI_API_KEY is not set")
    violations = report["violations"][:settings.EXPLAIN_MAX_ISSUES]
    if not violations:
        return {"model": settings.GEMINI_MODEL, "advisory": True, "truncated": False, "explanations": {}}
    raw = call_model(prompt_for(violations))
    try:
        parsed = TypeAdapter(dict[str, Explanation]).validate_json(raw)
    except ValidationError:
        raise HTTPException(502, "explanation service returned an unexpected format")
    allowed = {v["key"] for v in violations}
    return {"model": settings.GEMINI_MODEL, "advisory": True, "truncated": len(report["violations"]) > len(violations),
            "explanations": {k: v.model_dump() for k, v in parsed.items() if k in allowed}}
