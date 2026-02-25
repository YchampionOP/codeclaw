"""Tests for codeclaw.formatter — SFT-ready JSONL formatting."""

import json

import pytest

from codeclaw.formatter import format_session, format_sessions, write_jsonl


class TestFormatSession:
    def test_basic_format(self):
        session = {
            "session_id": "abc123",
            "model": "claude-sonnet-4-20250514",
            "git_branch": "main",
            "start_time": "2026-02-25T10:00:00Z",
            "project": "myproject",
            "trajectory_type": "sft_clean",
            "messages": [
                {"role": "user", "content": "Fix the bug"},
                {"role": "assistant", "content": "I'll fix it."},
            ],
        }
        result = format_session(session)
        assert "messages" in result
        assert "metadata" in result
        assert len(result["messages"]) == 2
        assert result["metadata"]["project"] == "myproject"
        assert result["metadata"]["trajectory_type"] == "sft_clean"
        assert result["metadata"]["session_id"] == "abc123"
        assert "claude-sonnet-4-20250514" in result["metadata"]["models_used"]

    def test_thinking_stripped_from_content(self):
        session = {
            "session_id": "abc",
            "messages": [
                {"role": "assistant", "content": "<thinking>hidden</thinking>visible text", "thinking": "hidden"},
            ],
        }
        result = format_session(session)
        assert result["messages"][0]["content"] == "visible text"
        assert "thinking_trace" in result["metadata"]

    def test_empty_messages(self):
        session = {"session_id": "abc", "messages": []}
        result = format_session(session)
        assert result["messages"] == []

    def test_defaults_for_missing_fields(self):
        session = {"messages": [{"role": "user", "content": "hi"}]}
        result = format_session(session)
        assert result["metadata"]["project"] == "unknown"
        assert result["metadata"]["trajectory_type"] == "sft_clean"


class TestFormatSessions:
    def test_multiple_sessions(self):
        sessions = [
            {"session_id": "a", "messages": [{"role": "user", "content": "hi"}]},
            {"session_id": "b", "messages": [{"role": "user", "content": "bye"}]},
        ]
        result = format_sessions(sessions)
        assert len(result) == 2


class TestWriteJsonl:
    def test_write_creates_file(self, tmp_path):
        output = tmp_path / "out.jsonl"
        sessions = [
            {"session_id": "a", "messages": [{"role": "user", "content": "hi"}]},
        ]
        count = write_jsonl(sessions, output)
        assert count == 1
        assert output.exists()
        lines = output.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert "messages" in data
        assert "metadata" in data

    def test_write_multiple(self, tmp_path):
        output = tmp_path / "out.jsonl"
        sessions = [
            {"session_id": "a", "messages": [{"role": "user", "content": "one"}]},
            {"session_id": "b", "messages": [{"role": "user", "content": "two"}]},
        ]
        count = write_jsonl(sessions, output)
        assert count == 2
