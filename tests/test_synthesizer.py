"""Tests for codeclaw.synthesizer — CODECLAW.md synthesizer."""

from pathlib import Path

import pytest

from codeclaw.synthesizer import (
    CODECLAW_MD_HEADER,
    MAX_LINES,
    _compute_dataset_health,
    _effective_tool_sequences,
    _extract_conventions,
    _extract_error_patterns,
    _extract_tool_sequences,
    synthesize,
)


# --- Sample data helpers ---

def _make_session(
    session_id: str = "sess-1",
    project: str = "myproject",
    trajectory_type: str = "sft_clean",
    messages: list | None = None,
) -> dict:
    if messages is None:
        messages = []
    return {
        "session_id": session_id,
        "project": project,
        "trajectory_type": trajectory_type,
        "model": "claude-3",
        "messages": messages,
    }


def _make_assistant_msg(content: str = "", tool_names: list[str] | None = None) -> dict:
    tool_uses = [{"tool": t, "input": ""} for t in (tool_names or [])]
    return {"role": "assistant", "content": content, "tool_uses": tool_uses}


def _make_user_msg(content: str = "") -> dict:
    return {"role": "user", "content": content, "tool_uses": []}


# --- _extract_tool_sequences ---

class TestExtractToolSequences:
    def test_empty_sessions(self):
        assert _extract_tool_sequences([]) == []

    def test_no_tools(self):
        session = _make_session(messages=[_make_user_msg("hello")])
        assert _extract_tool_sequences([session]) == []

    def test_single_tool(self):
        session = _make_session(messages=[_make_assistant_msg(tool_names=["Read"])])
        seqs = _extract_tool_sequences([session])
        assert seqs == [["Read"]]

    def test_multiple_tools(self):
        session = _make_session(messages=[
            _make_assistant_msg(tool_names=["Read", "Bash"]),
            _make_assistant_msg(tool_names=["Write"]),
        ])
        seqs = _extract_tool_sequences([session])
        assert seqs == [["Read", "Bash", "Write"]]

    def test_multiple_sessions(self):
        s1 = _make_session(session_id="s1", messages=[_make_assistant_msg(tool_names=["Read"])])
        s2 = _make_session(session_id="s2", messages=[_make_assistant_msg(tool_names=["Bash"])])
        seqs = _extract_tool_sequences([s1, s2])
        assert seqs == [["Read"], ["Bash"]]


# --- _extract_error_patterns ---

class TestExtractErrorPatterns:
    def test_no_errors(self):
        session = _make_session(messages=[_make_user_msg("all good")])
        counter = _extract_error_patterns([session])
        assert len(counter) == 0

    def test_error_in_user_message(self):
        session = _make_session(messages=[_make_user_msg("Error: file not found")])
        counter = _extract_error_patterns([session])
        assert len(counter) >= 1

    def test_deduplication_within_session(self):
        msg = _make_user_msg("Error: missing module")
        session = _make_session(messages=[msg, msg])  # same message twice
        counter = _extract_error_patterns([session])
        # Should only count once per session
        assert counter.get("Error: missing module", 0) <= 1

    def test_cross_session_counting(self):
        msg = _make_user_msg("Error: something broke")
        s1 = _make_session(session_id="s1", messages=[msg])
        s2 = _make_session(session_id="s2", messages=[msg])
        counter = _extract_error_patterns([s1, s2])
        key = "Error: something broke"
        assert counter.get(key, 0) == 2


# --- _extract_conventions ---

class TestExtractConventions:
    def test_no_conventions(self):
        session = _make_session(messages=[_make_user_msg("just a user message")])
        convs = _extract_conventions([session])
        assert convs == []

    def test_single_convention_not_repeated(self):
        msg = _make_assistant_msg("Always use type hints in Python")
        session = _make_session(messages=[msg])
        convs = _extract_conventions([session])
        # Single occurrence — should NOT appear (needs >= 2 sessions)
        assert convs == []

    def test_convention_repeated_across_sessions(self):
        msg = _make_assistant_msg("Always use type hints in Python functions")
        s1 = _make_session(session_id="s1", messages=[msg])
        s2 = _make_session(session_id="s2", messages=[msg])
        convs = _extract_conventions([s1, s2])
        assert len(convs) >= 1
        assert any("type hints" in c.lower() for c in convs)


# --- _effective_tool_sequences ---

class TestEffectiveToolSequences:
    def test_empty(self):
        assert _effective_tool_sequences([]) == []

    def test_single_sequence_no_bigram(self):
        seqs = [["Read"]]
        result = _effective_tool_sequences(seqs)
        assert result == []

    def test_bigram_below_threshold(self):
        seqs = [["Read", "Bash"]]  # only once — threshold is 2
        result = _effective_tool_sequences(seqs)
        assert result == []

    def test_bigram_above_threshold(self):
        seqs = [["Read", "Bash"], ["Read", "Bash"], ["Read", "Bash"]]
        result = _effective_tool_sequences(seqs)
        assert ("Read", "Bash") in result


# --- _compute_dataset_health ---

class TestComputeDatasetHealth:
    def test_empty(self):
        health = _compute_dataset_health([])
        assert health["total_sessions"] == 0
        assert health["trajectory_counts"] == {}

    def test_counts(self):
        sessions = [
            _make_session(session_id="s1", trajectory_type="sft_clean"),
            _make_session(session_id="s2", trajectory_type="sft_clean"),
            _make_session(session_id="s3", trajectory_type="debugging_trace"),
        ]
        health = _compute_dataset_health(sessions)
        assert health["total_sessions"] == 3
        assert health["trajectory_counts"]["sft_clean"] == 2
        assert health["trajectory_counts"]["debugging_trace"] == 1


# --- synthesize ---

class TestSynthesize:
    def test_creates_file(self, tmp_path):
        sessions = [_make_session()]
        out = synthesize(sessions, "myproject", tmp_path)
        assert out.exists()
        assert out.name == "CODECLAW.md"

    def test_header_present(self, tmp_path):
        sessions = [_make_session()]
        out = synthesize(sessions, "myproject", tmp_path)
        content = out.read_text()
        assert CODECLAW_MD_HEADER in content

    def test_project_name_in_header(self, tmp_path):
        sessions = [_make_session()]
        out = synthesize(sessions, "myproject", tmp_path)
        content = out.read_text()
        assert "myproject" in content

    def test_session_count_in_health(self, tmp_path):
        sessions = [
            _make_session(session_id="s1"),
            _make_session(session_id="s2"),
        ]
        out = synthesize(sessions, "myproject", tmp_path)
        content = out.read_text()
        assert "2" in content  # total sessions shown

    def test_max_lines_respected(self, tmp_path):
        # Create many sessions with lots of errors to inflate line count
        messages = [_make_user_msg(f"Error: thing {i} broke badly in module foo") for i in range(200)]
        sessions = [_make_session(session_id=f"s{i}", messages=messages[:2]) for i in range(50)]
        out = synthesize(sessions, "bigproject", tmp_path)
        content = out.read_text()
        assert len(content.splitlines()) <= MAX_LINES

    def test_recurring_bugs_section(self, tmp_path):
        msg = _make_user_msg("Error: database connection failed")
        sessions = [
            _make_session(session_id=f"s{i}", messages=[msg]) for i in range(3)
        ]
        out = synthesize(sessions, "myproject", tmp_path)
        content = out.read_text()
        assert "Recurring Bugs" in content

    def test_effective_tool_sequences_section(self, tmp_path):
        sessions = [
            _make_session(
                session_id=f"s{i}",
                messages=[_make_assistant_msg(tool_names=["Read", "Bash"])],
            )
            for i in range(3)
        ]
        out = synthesize(sessions, "myproject", tmp_path)
        content = out.read_text()
        assert "Tool-Call Sequences" in content

    def test_overwrites_existing(self, tmp_path):
        sessions = [_make_session()]
        out = synthesize(sessions, "myproject", tmp_path)
        first_content = out.read_text()

        sessions2 = [_make_session(session_id="s2"), _make_session(session_id="s3")]
        out2 = synthesize(sessions2, "myproject", tmp_path)
        second_content = out2.read_text()

        assert first_content != second_content
        assert "2" in second_content  # reflects new session count
