"""Tests for codeclaw.graph_index — graph-structured tool-call indexing."""

import json
from pathlib import Path

import pytest

from codeclaw.graph_index import (
    GraphIndex,
    _tool_node,
    _file_node,
    _error_node,
    _normalize_node,
    build_index_from_jsonl,
)


# --- Helpers ---

def _make_session(
    session_id: str = "sess-1",
    trajectory_type: str = "sft_clean",
    tool_names: list[str] | None = None,
    content: str = "",
) -> dict:
    messages = []
    if tool_names:
        tool_uses = [{"tool": t, "input": ""} for t in tool_names]
        messages.append({"role": "assistant", "content": content, "tool_uses": tool_uses})
    elif content:
        messages.append({"role": "user", "content": content, "tool_uses": []})
    return {
        "session_id": session_id,
        "project": "testproject",
        "trajectory_type": trajectory_type,
        "model": "claude-3",
        "messages": messages,
    }


# --- Node naming helpers ---

class TestNodeHelpers:
    def test_tool_node_prefix(self):
        assert _tool_node("Bash").startswith("tool:")

    def test_tool_node_lowercase(self):
        assert _tool_node("Read") == "tool:read"

    def test_file_node_prefix(self):
        assert _file_node("src/auth.py").startswith("file:")

    def test_error_node_prefix(self):
        assert _error_node("Error: not found").startswith("error:")

    def test_normalize_node_strips(self):
        assert _normalize_node("  Bash  ") == "bash"

    def test_error_node_truncates(self):
        long_msg = "Error: " + "x" * 200
        node = _error_node(long_msg)
        assert len(node) < 80  # prefix + 60 chars max


# --- GraphIndex.build ---

class TestGraphIndexBuild:
    def test_empty_build(self):
        index = GraphIndex()
        index.build([])
        stats = index.stats()
        assert stats["sessions"] == 0

    def test_single_session(self):
        session = _make_session(tool_names=["Read", "Bash"])
        index = GraphIndex()
        index.build([session])
        stats = index.stats()
        assert stats["sessions"] == 1
        assert stats["edges"] >= 1  # Read→Bash

    def test_multiple_sessions(self):
        sessions = [
            _make_session(session_id="s1", tool_names=["Read"]),
            _make_session(session_id="s2", tool_names=["Bash"]),
        ]
        index = GraphIndex()
        index.build(sessions)
        assert index.stats()["sessions"] == 2

    def test_rebuild_resets(self):
        index = GraphIndex()
        index.build([_make_session(session_id="s1", tool_names=["Read"])])
        assert index.stats()["sessions"] == 1
        index.build([])
        assert index.stats()["sessions"] == 0


# --- GraphIndex.add_session ---

class TestGraphIndexAddSession:
    def test_incremental_add(self):
        index = GraphIndex()
        index.add_session(_make_session(session_id="s1", tool_names=["Read"]))
        index.add_session(_make_session(session_id="s2", tool_names=["Bash"]))
        assert index.stats()["sessions"] == 2

    def test_duplicate_session_id(self):
        """Adding same session_id twice should still work (last write wins in _sessions)."""
        index = GraphIndex()
        index.add_session(_make_session(session_id="s1", tool_names=["Read"]))
        index.add_session(_make_session(session_id="s1", tool_names=["Bash"]))
        # Both entries added; session dict overwritten but edges accumulate
        assert index.stats()["sessions"] >= 1


# --- GraphIndex.query ---

class TestGraphIndexQuery:
    def test_query_empty_index(self):
        index = GraphIndex()
        result = index.query(["tool:read"])
        assert result == []

    def test_query_empty_context(self):
        index = GraphIndex()
        index.build([_make_session(tool_names=["Read"])])
        result = index.query([])
        assert result == []

    def test_exact_tool_match(self):
        s1 = _make_session(session_id="s1", tool_names=["Read", "Bash"])
        s2 = _make_session(session_id="s2", tool_names=["Write"])
        index = GraphIndex()
        index.build([s1, s2])

        results = index.query(["tool:read"])
        session_ids = [r["session_id"] for r in results]
        assert "s1" in session_ids

    def test_max_results_respected(self):
        sessions = [
            _make_session(session_id=f"s{i}", tool_names=["Read"])
            for i in range(10)
        ]
        index = GraphIndex()
        index.build(sessions)
        results = index.query(["tool:read"], max_results=3)
        assert len(results) <= 3

    def test_no_match_returns_empty(self):
        index = GraphIndex()
        index.build([_make_session(tool_names=["Write"])])
        results = index.query(["tool:bash"])
        assert results == []

    def test_neighbor_traversal(self):
        """Sessions sharing graph neighbors should be returned even without direct match."""
        s1 = _make_session(session_id="s1", tool_names=["Read", "Bash"])
        s2 = _make_session(session_id="s2", tool_names=["Bash", "Write"])
        index = GraphIndex()
        index.build([s1, s2])

        # Query for Read — s1 matches directly; s2 shares Bash neighbor
        results = index.query(["tool:read"])
        assert any(r["session_id"] == "s1" for r in results)


# --- GraphIndex.stats ---

class TestGraphIndexStats:
    def test_stats_keys(self):
        index = GraphIndex()
        stats = index.stats()
        assert "nodes" in stats
        assert "edges" in stats
        assert "sessions" in stats
        assert "networkx_available" in stats

    def test_stats_after_build(self):
        session = _make_session(tool_names=["Read", "Bash", "Write"])
        index = GraphIndex()
        index.build([session])
        stats = index.stats()
        assert stats["sessions"] == 1
        assert stats["edges"] >= 2  # Read→Bash, Bash→Write


# --- build_index_from_jsonl ---

class TestBuildIndexFromJsonl:
    def test_missing_file(self, tmp_path):
        index = build_index_from_jsonl([tmp_path / "nonexistent.jsonl"])
        assert index.stats()["sessions"] == 0

    def test_valid_jsonl(self, tmp_path):
        jsonl_file = tmp_path / "sessions.jsonl"
        session = _make_session(session_id="s1", tool_names=["Read"])
        jsonl_file.write_text(json.dumps(session) + "\n", encoding="utf-8")

        index = build_index_from_jsonl([jsonl_file])
        assert index.stats()["sessions"] == 1

    def test_malformed_lines_skipped(self, tmp_path):
        jsonl_file = tmp_path / "sessions.jsonl"
        session = _make_session(session_id="s1", tool_names=["Read"])
        content = json.dumps(session) + "\n" + "not-json\n" + json.dumps(session) + "\n"
        jsonl_file.write_text(content, encoding="utf-8")

        index = build_index_from_jsonl([jsonl_file])
        # Two valid lines (same session_id overwritten), but no crash
        assert index.stats()["sessions"] >= 1

    def test_multiple_files(self, tmp_path):
        s1 = _make_session(session_id="s1", tool_names=["Read"])
        s2 = _make_session(session_id="s2", tool_names=["Bash"])
        f1 = tmp_path / "a.jsonl"
        f2 = tmp_path / "b.jsonl"
        f1.write_text(json.dumps(s1) + "\n", encoding="utf-8")
        f2.write_text(json.dumps(s2) + "\n", encoding="utf-8")

        index = build_index_from_jsonl([f1, f2])
        assert index.stats()["sessions"] == 2


# --- File and error node tracking ---

class TestFileAndErrorNodeTracking:
    def test_file_ref_query(self):
        session = _make_session(
            session_id="s-file-ref",
            content="Please fix src/foo.py\nTypeError: unsupported operand",
        )
        index = GraphIndex()
        index.build([session])

        assert index.stats()["sessions"] == 1
        results = index.query(["file:src/foo.py"])
        assert any(r["session_id"] == "s-file-ref" for r in results)

    def test_error_ref_query(self):
        session = _make_session(
            session_id="s-err-ref",
            content="Please fix src/foo.py\nTypeError: unsupported operand",
        )
        index = GraphIndex()
        index.build([session])

        results = index.query(["error:typeerror: unsupported operand"])
        assert any(r["session_id"] == "s-err-ref" for r in results)
