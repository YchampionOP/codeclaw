"""Session trajectory classifier heuristics."""


def classify_trajectory(session: dict) -> str:
    """
    Assign trajectory_type to a session based on message content heuristics.
    Returns one of: correction_loop | debugging_trace | iterative_build | refactor | sft_clean
    """
    messages = session.get("messages", [])

    CORRECTION_SIGNALS = [
        "wrong",
        "error",
        "that's not",
        "fix",
        "broken",
        "failed",
        "doesn't work",
        "incorrect",
        "bug",
        "not what i",
        "that won't",
        "not right",
    ]
    DEBUG_TOOLS = ["bash", "python", "execute"]
    REFACTOR_SIGNALS = ["refactor", "clean up", "rewrite", "simplify", "restructure", "reorganize", "consolidate"]

    # Check for correction loop: user message after assistant that contains correction signal
    for i, msg in enumerate(messages):
        if msg.get("role") == "user" and i > 0:
            content = str(msg.get("content", "")).lower()
            if any(sig in content for sig in CORRECTION_SIGNALS):
                return "correction_loop"

    # Check for debugging trace: tool uses with bash + error outputs
    has_bash = any(
        any(str(t.get("tool", "")).lower() in DEBUG_TOOLS for t in msg.get("tool_uses", []))
        for msg in messages if msg.get("role") == "assistant"
    )
    has_error_output = any(
        "error" in str(msg.get("content", "")).lower()
        or "traceback" in str(msg.get("content", "")).lower()
        for msg in messages
    )
    if has_bash and has_error_output:
        return "debugging_trace"

    # Refactor
    first_user = next((m for m in messages if m.get("role") == "user"), None)
    if first_user:
        content = str(first_user.get("content", "")).lower()
        if any(sig in content for sig in REFACTOR_SIGNALS):
            return "refactor"

    # Iterative build: long sessions with tool use but no corrections
    if len(messages) > 8 and has_bash:
        return "iterative_build"

    return "sft_clean"
