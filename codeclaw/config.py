"""Persistent config for CodeClaw — stored at ~/.codeclaw/config.json"""

import json
import sys
from pathlib import Path
from typing import TypedDict

CONFIG_DIR = Path.home() / ".codeclaw"
CONFIG_FILE = CONFIG_DIR / "config.json"
LEGACY_CONFIG_FILE = Path.home() / ".codeclaw" / "config.json"


class CodeClawConfig(TypedDict, total=False):
    """Expected shape of the config dict."""

    repo: str | None
    source: str | None  # "claude" | "codex" | "both"
    excluded_projects: list[str]
    redact_strings: list[str]
    redact_usernames: list[str]
    last_export: dict
    stage: str | None  # "auth" | "configure" | "review" | "confirmed" | "done"
    projects_confirmed: bool  # True once user has addressed folder exclusions
    watch_interval_seconds: int
    min_sessions_before_push: int
    auto_push: bool
    last_synced_at: str | None
    synced_session_ids: list[str]
    publish_attestation: str | None
    dataset_enabled: bool
    disabled_projects: list[str]
    stats_total_exports: int
    stats_total_publishes: int
    stats_total_exported_sessions: int
    stats_total_redactions: int
    stats_total_input_tokens: int
    stats_total_output_tokens: int


DEFAULT_CONFIG: CodeClawConfig = {
    "repo": None,
    "source": None,
    "excluded_projects": [],
    "redact_strings": [],
    "synced_session_ids": [],
    "watch_interval_seconds": 60,
    "min_sessions_before_push": 5,
    "auto_push": False,
}


def load_config() -> CodeClawConfig:
    source = CONFIG_FILE if CONFIG_FILE.exists() else LEGACY_CONFIG_FILE
    if source.exists():
        try:
            with open(source) as f:
                stored = json.load(f)
            return {**DEFAULT_CONFIG, **stored}
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: could not read {source}: {e}", file=sys.stderr)
    return dict(DEFAULT_CONFIG)


def save_config(config: CodeClawConfig) -> None:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        CONFIG_FILE.chmod(0o600)
    except OSError as e:
        print(f"Warning: could not save {CONFIG_FILE}: {e}", file=sys.stderr)
