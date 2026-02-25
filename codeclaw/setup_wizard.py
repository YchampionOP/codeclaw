"""First-run interactive setup wizard for CodeClaw."""

import getpass
import re
from pathlib import Path

from .config import CodeClawConfig, load_config, save_config
from .publisher import ensure_dataset_exists, validate_token

BANNER = r"""
   ____          _       ____ _
  / ___|___   __| | ___ / ___| | __ ___      __
 | |   / _ \ / _` |/ _ \ |   | |/ _` \ \ /\ / /
 | |__| (_) | (_| |  __/ |___| | (_| |\ V  V /
  \____\___/ \__,_|\___|\____|_|\__,_| \_/\_/

  CodeClaw — Claude Code → Training Data Pipeline
"""


def _parse_dataset_repo(raw: str) -> str | None:
    """Parse and validate a dataset repo identifier.

    Accepts either 'username/dataset-name' or a full HuggingFace URL like
    'https://huggingface.co/datasets/username/dataset-name'.
    """
    raw = raw.strip()
    if not raw:
        return None

    # Extract from full URL
    url_match = re.match(
        r"https?://huggingface\.co/datasets/([^/]+/[^/\s]+)", raw
    )
    if url_match:
        return url_match.group(1).rstrip("/")

    # Validate plain repo format: exactly one "/"
    if raw.count("/") == 1 and not raw.startswith("/") and not raw.endswith("/"):
        return raw

    return None


def _prompt_dataset_repo() -> str:
    """Prompt for HuggingFace dataset repo until valid."""
    while True:
        raw = input("Enter your HuggingFace dataset repo (e.g. username/cc-logs): ")
        parsed = _parse_dataset_repo(raw)
        if parsed:
            return parsed
        print("  ✗ Invalid format. Must be 'username/dataset-name' or a HuggingFace URL.")


def _prompt_hf_token() -> str:
    """Prompt for HuggingFace token until valid."""
    while True:
        token = getpass.getpass("Enter your HuggingFace token (needs write access) [input hidden]: ")
        token = token.strip()
        if not token:
            print("  ✗ Token cannot be empty.")
            continue
        if validate_token(token):
            return token
        print("  ✗ Token validation failed. Please check your token and try again.")


def _prompt_logs_dir() -> str:
    """Prompt for Claude Code logs directory."""
    default = "~/.claude/projects"
    raw = input(f"Claude Code logs directory [{default}]: ").strip()
    if not raw:
        raw = default
    expanded = Path(raw).expanduser()
    if not expanded.is_dir():
        print(f"  ⚠ Directory {expanded} does not exist. Using anyway.")
    return raw


def _prompt_yes_no(prompt: str, default: bool = True) -> bool:
    """Prompt for a yes/no answer."""
    suffix = "(y/n) [y]" if default else "(y/n) [n]"
    raw = input(f"{prompt} {suffix}: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def run_wizard(force: bool = False) -> CodeClawConfig:
    """Run the interactive setup wizard.

    Args:
        force: If True, re-run even if config already exists.

    Returns:
        The saved config dict.
    """
    print(BANNER)

    dataset_repo = _prompt_dataset_repo()
    hf_token = _prompt_hf_token()
    claude_logs_dir = _prompt_logs_dir()
    auto_push = _prompt_yes_no("Auto-push after each sync?")
    private = _prompt_yes_no("Make dataset private?")

    config = load_config()
    config["dataset_repo"] = dataset_repo
    config["hf_token"] = hf_token
    config["claude_logs_dir"] = claude_logs_dir
    config["auto_push"] = auto_push
    config["private"] = private
    if "last_synced_at" not in config:
        config["last_synced_at"] = None
    if "synced_session_ids" not in config:
        config["synced_session_ids"] = []

    save_config(config)

    # Create dataset on HuggingFace
    try:
        ensure_dataset_exists(dataset_repo, hf_token, private)
        print(f"  ✓ Dataset '{dataset_repo}' is ready on HuggingFace.")
    except Exception as e:
        print(f"  ⚠ Could not create dataset: {e}")

    print("\n✓ Setup complete. Run `codeclaw sync` to push your first sessions.")
    return config


def ensure_setup() -> CodeClawConfig:
    """Ensure config exists; run wizard if not."""
    config = load_config()
    if config.get("dataset_repo") and config.get("hf_token"):
        return config
    return run_wizard()
