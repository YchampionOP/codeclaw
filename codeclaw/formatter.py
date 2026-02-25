"""Convert sessions to SFT-ready JSONL format for CodeClaw."""

import json
import re
from pathlib import Path

# Pattern to match <thinking>...</thinking> blocks
_THINKING_TAG_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL)


def format_session(session: dict) -> dict:
    """Convert a parsed session into SFT-ready format.

    Output format:
    {
      "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ],
      "metadata": {
        "project": "...",
        "trajectory_type": "...",
        "models_used": [...],
        "session_id": "...",
        "git_branch": "...",
        "start_time": "..."
      }
    }
    """
    sft_messages = []
    models_used = set()
    thinking_traces = []

    for msg in session.get("messages", []):
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant":
            # Strip thinking tags from content, store separately
            thinking = msg.get("thinking", "")
            if thinking:
                thinking_traces.append(thinking)

            # Remove <thinking>...</thinking> from content if present
            if content:
                content = _THINKING_TAG_RE.sub("", content).strip()

        if content:
            sft_messages.append({"role": role, "content": content})

    model = session.get("model")
    if model:
        models_used.add(model)

    metadata = {
        "project": session.get("project", "unknown"),
        "trajectory_type": session.get("trajectory_type", "sft_clean"),
        "models_used": sorted(models_used),
        "session_id": session.get("session_id", ""),
        "git_branch": session.get("git_branch"),
        "start_time": session.get("start_time"),
    }

    if thinking_traces:
        metadata["thinking_trace"] = "\n\n".join(thinking_traces)

    return {
        "messages": sft_messages,
        "metadata": metadata,
    }


def format_sessions(sessions: list[dict]) -> list[dict]:
    """Format multiple sessions into SFT-ready dicts."""
    return [format_session(s) for s in sessions]


def write_jsonl(sessions: list[dict], output_path: Path) -> int:
    """Write formatted sessions to a JSONL file.

    Returns the number of sessions written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w") as f:
        for session in sessions:
            formatted = format_session(session) if "metadata" not in session else session
            f.write(json.dumps(formatted) + "\n")
            count += 1
    return count
