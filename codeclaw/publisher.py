"""HuggingFace push logic for CodeClaw."""

import json
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def ensure_dataset_exists(dataset_repo: str, hf_token: str, private: bool = True) -> None:
    """Create the HF dataset repo if it doesn't exist, with an initial README."""
    api = HfApi(token=hf_token)
    api.create_repo(
        repo_id=dataset_repo,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    # Create initial README if not present
    try:
        api.hf_hub_download(
            repo_id=dataset_repo,
            filename="README.md",
            repo_type="dataset",
        )
    except Exception:
        readme_content = _build_initial_readme(dataset_repo)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(readme_content)
            tmp_path = f.name
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=dataset_repo,
            repo_type="dataset",
        )
        Path(tmp_path).unlink(missing_ok=True)


def push_jsonl(
    jsonl_path: Path,
    dataset_repo: str,
    hf_token: str,
    path_in_repo: str = "data/train.jsonl",
) -> None:
    """Push a JSONL file to the HuggingFace dataset repo."""
    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=str(jsonl_path),
        path_in_repo=path_in_repo,
        repo_id=dataset_repo,
        repo_type="dataset",
    )


def validate_token(hf_token: str) -> bool:
    """Validate a HuggingFace token by calling whoami()."""
    try:
        api = HfApi(token=hf_token)
        api.whoami()
        return True
    except Exception:
        return False


def _build_initial_readme(dataset_repo: str) -> str:
    return f"""---
tags:
- logs
- claude-code
- codeclaw
---

# {dataset_repo}

Dataset exported with [CodeClaw](https://github.com/ychampion/codeclaw).

**Tag: `codeclaw`** — [Browse all CodeClaw datasets](https://huggingface.co/datasets?other=codeclaw)
"""
