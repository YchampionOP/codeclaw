"""Sensitive info removal for CodeClaw — wraps secrets and anonymizer modules."""

from .anonymizer import Anonymizer
from .secrets import redact_custom_strings, redact_session, redact_text, scan_text

__all__ = [
    "Anonymizer",
    "redact_text",
    "redact_session",
    "redact_custom_strings",
    "scan_text",
    "redact_all_sessions",
]


def redact_all_sessions(
    sessions: list[dict],
    custom_strings: list[str] | None = None,
) -> tuple[list[dict], int]:
    """Redact secrets from a list of sessions.

    Returns:
        Tuple of (redacted sessions, total redaction count).
    """
    total_redactions = 0
    redacted = []
    for session in sessions:
        session, count = redact_session(session, custom_strings)
        total_redactions += count
        redacted.append(session)
    return redacted, total_redactions
