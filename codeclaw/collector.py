"""Collect and structure Claude Code session logs for CodeClaw."""

from pathlib import Path

from .anonymizer import Anonymizer
from .config import CodeClawConfig
from .parser import (
    CLAUDE_SOURCE,
    PROJECTS_DIR,
    discover_projects,
    parse_project_sessions,
)


def collect_new_sessions(
    config: CodeClawConfig,
    source_filter: str = "auto",
) -> list[dict]:
    """Collect sessions not yet synced, based on config tracking.

    Walks the Claude/Codex projects directory, parses JSONL logs, and returns
    only sessions whose IDs are not in config['synced_session_ids'].
    """
    synced_ids = set(config.get("synced_session_ids", []))
    logs_dir = config.get("claude_logs_dir", "~/.claude/projects")
    logs_path = Path(logs_dir).expanduser()

    # Discover available projects
    projects = discover_projects()

    if source_filter != "auto":
        projects = [p for p in projects if p["source"] == source_filter]

    anonymizer = Anonymizer(
        extra_usernames=config.get("redact_usernames"),
    )

    all_sessions = []
    for project in projects:
        # Skip excluded projects
        if project["dir_name"] in config.get("excluded_projects", []):
            continue

        sessions = parse_project_sessions(
            project["dir_name"],
            anonymizer=anonymizer,
            source=project["source"],
        )
        for session in sessions:
            session_id = session.get("session_id", "")
            if session_id and session_id not in synced_ids:
                session["project"] = project.get("display_name", project["dir_name"])
                all_sessions.append(session)

    return all_sessions


def count_pending_sessions(config: CodeClawConfig) -> int:
    """Count how many sessions haven't been synced yet."""
    return len(collect_new_sessions(config))
