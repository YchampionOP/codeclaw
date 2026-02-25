CodeClaw -- Code Intelligence

What This Repo Does
CodeClaw collects Claude Code session logs, redacts secrets, classifies trajectories,
and pushes structured training data to HuggingFace automatically.

Rules For Claude Code Working On This Repo
Never hardcode dataset repo IDs -- always read from ~/.codeclaw/config.json

When adding redaction patterns, update BOTH codeclaw/secrets.py AND document in this file

The daemon MUST NEVER block or slow the user's active Claude Code session

Auto-push in daemon mode bypasses attestation gates -- this is intentional for private use

After any change to classifier.py, run: python -m pytest tests/ -v

Config at ~/.codeclaw/config.json, always chmod 600, always use .get() with defaults

Adding New Redaction Patterns
Add to codeclaw/secrets.py in the REDACT_PATTERNS dict.
Also add to ~/.codeclaw/blocklist.txt (one per line) for user-defined patterns.

Trajectory Types
correction_loop: user corrects assistant -> HIGH training value

debugging_trace: bash+errors -> HIGH training value

iterative_build: long multi-tool session -> MEDIUM value

refactor: user asks to clean/rewrite -> MEDIUM value

sft_clean: clean first-try solution -> MEDIUM value, filter if < 4 turns

text
