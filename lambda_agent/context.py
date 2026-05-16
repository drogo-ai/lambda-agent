"""
Context Management Module
=========================
Keeps the agent's context window lean using two complementary strategies:

1. **Full Transcript** (``.agent/transcript.jsonl``)
   Append-only log of every tool call and response at full length.
   This is the ground-truth record and is never truncated.

2. **Sliding-window trimmer** (``trim_chat_history``)
   After each turn, older tool-call responses in the live chat history
   are truncated so the model's prompt stays within budget.

   Window tiers (counted from most-recent tool response):
     Tier 1  — last 4 responses   → up to 500 chars each
     Tier 2  — next 8 responses   → up to 180 chars each
     Tier 3  — anything older     → up to 80  chars each
"""

import json
import os
from datetime import datetime

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AGENT_DIR = ".agent"
TRANSCRIPT_FILE = os.path.join(AGENT_DIR, "transcript.jsonl")


def clip(text: str, max_chars: int) -> str:
    """Truncate *text* to *max_chars*.

    If the text is clipped, a notice is appended so the model knows
    the response was shortened.
    """
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[TRUNCATED — original {len(text)} chars]"


# ---------------------------------------------------------------------------
# Full transcript (append-only log — never truncated)
# ---------------------------------------------------------------------------


class Transcript:
    """Append-only JSONL log of every exchange in the session."""

    def __init__(self):
        os.makedirs(AGENT_DIR, exist_ok=True)
        self._path = os.path.abspath(TRANSCRIPT_FILE)

    def log(self, role: str, content: str, meta: dict | None = None):
        """Append a single entry to the transcript file.

        Args:
            role: One of 'user', 'assistant', 'tool_call', 'tool_result'.
            content: The full, untruncated payload.
            meta: Optional dict of extra metadata (tool name, args, etc.).
        """
        entry: dict = {
            "ts": datetime.now().isoformat(),
            "role": role,
            "content": content,
        }
        if meta:
            entry["meta"] = meta
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass  # Transcript logging must never crash the agent


# ---------------------------------------------------------------------------
# Sliding-window trimmer
# ---------------------------------------------------------------------------

# Default tier settings
TIER1_COUNT = 4  # most recent N tool responses
TIER1_LIMIT = None  # chars to keep (None means do not truncate)

TIER2_COUNT = 8  # next N tool responses
TIER2_LIMIT = 180

TIER3_LIMIT = 80  # everything older


def trim_chat_history(
    history: list,
    tier1_count: int = TIER1_COUNT,
    tier1_limit: int | None = TIER1_LIMIT,
    tier2_count: int = TIER2_COUNT,
    tier2_limit: int = TIER2_LIMIT,
    tier3_limit: int = TIER3_LIMIT,
) -> None:
    """Mutate *history* in-place, truncating function-response payloads.

    Works directly on the Gemini SDK's ``_curated_history`` list
    (a list of ``Content`` objects whose ``parts`` may contain
    ``FunctionResponse`` items).

    The most recent *tier1_count* function responses are kept at
    *tier1_limit* chars; the next *tier2_count* at *tier2_limit*;
    anything older is clipped to *tier3_limit*.
    """
    # Collect every (content_index, part_index) that holds a function_response
    fr_locations: list[tuple[int, int]] = []

    for ci, content in enumerate(history):
        parts = getattr(content, "parts", None) or []
        for pi, part in enumerate(parts):
            fn_resp = getattr(part, "function_response", None)
            if fn_resp is not None:
                fr_locations.append((ci, pi))

    if not fr_locations:
        return

    # Walk from most-recent → oldest and apply the right tier limit
    for rank, (ci, pi) in enumerate(reversed(fr_locations)):
        part = history[ci].parts[pi]
        resp = part.function_response.response

        if resp is None or "result" not in resp:
            continue

        original = str(resp["result"])

        if rank < tier1_count:
            limit = tier1_limit
        elif rank < tier1_count + tier2_count:
            limit = tier2_limit
        else:
            limit = tier3_limit

        if limit is not None:
            resp["result"] = clip(original, limit)
