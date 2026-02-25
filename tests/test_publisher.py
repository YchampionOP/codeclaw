"""Tests for codeclaw.publisher — HuggingFace push logic."""

import pytest

from codeclaw.publisher import _build_initial_readme


class TestBuildInitialReadme:
    def test_contains_repo_name(self):
        readme = _build_initial_readme("user/my-dataset")
        assert "user/my-dataset" in readme

    def test_contains_yaml_header(self):
        readme = _build_initial_readme("user/ds")
        assert "---" in readme
        assert "tags:" in readme
        assert "codeclaw" in readme

    def test_contains_link(self):
        readme = _build_initial_readme("user/ds")
        assert "CodeClaw" in readme
