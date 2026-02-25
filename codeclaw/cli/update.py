"""Update-skill and synthesize subcommand helpers."""

import json
import sys
import urllib.request
from pathlib import Path

from ._helpers import SKILL_URL


def update_skill(target: str) -> None:
    """Download and install the codeclaw skill for a coding agent."""
    if target != "claude":
        print(f"Error: unknown target '{target}'. Supported: claude", file=sys.stderr)
        sys.exit(1)

    dest = Path.cwd() / ".claude" / "skills" / "codeclaw" / "SKILL.md"
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading skill from {SKILL_URL}...")
    try:
        with urllib.request.urlopen(SKILL_URL, timeout=15) as resp:
            content = resp.read().decode()
    except (OSError, urllib.error.URLError) as e:
        print(f"Error downloading skill: {e}", file=sys.stderr)
        # Fall back to bundled copy
        bundled = Path(__file__).resolve().parent.parent.parent / "docs" / "SKILL.md"
        if bundled.exists():
            print(f"Using bundled copy from {bundled}")
            content = bundled.read_text()
        else:
            print("No bundled copy available either.", file=sys.stderr)
            sys.exit(1)

    dest.write_text(content)
    print(f"Skill installed to {dest}")
    print(json.dumps({
        "installed": str(dest),
        "next_steps": ["Run: codeclaw prep"],
        "next_command": "codeclaw prep",
    }, indent=2))


def _handle_synthesize(args) -> None:
    from ..synthesizer import synthesize_for_project

    project = args.project
    output_dir = args.output if args.output else None
    out_path = synthesize_for_project(project, jsonl_dir=output_dir)
    if out_path is None:
        print(json.dumps({"error": f"No sessions found for project '{project}'."}, indent=2))
        sys.exit(1)
    print(json.dumps({"synthesized": str(out_path), "project": project}, indent=2))
