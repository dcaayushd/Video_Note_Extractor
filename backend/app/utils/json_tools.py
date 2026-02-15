from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    if raw_text.startswith("{"):
        return json.loads(raw_text)

    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1))

    object_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    if object_match:
        return json.loads(object_match.group(1))

    raise ValueError("No JSON object found in model response.")

