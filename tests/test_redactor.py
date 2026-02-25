"""Tests for codeclaw.redactor — sensitive info removal wrapper."""

import pytest

from codeclaw.redactor import redact_all_sessions


class TestRedactAllSessions:
    def test_empty_list(self):
        sessions, count = redact_all_sessions([])
        assert sessions == []
        assert count == 0

    def test_no_secrets(self):
        sessions = [
            {"messages": [{"role": "user", "content": "hello world"}]},
        ]
        result, count = redact_all_sessions(sessions)
        assert len(result) == 1
        assert count == 0

    def test_with_secret(self):
        sessions = [
            {"messages": [{"role": "user", "content": "my key is sk-ant-AAAAAAAAAAAAAAAAAAAAAA"}]},
        ]
        result, count = redact_all_sessions(sessions)
        assert count > 0
        assert "[REDACTED]" in result[0]["messages"][0]["content"]

    def test_custom_strings(self):
        sessions = [
            {"messages": [{"role": "user", "content": "contact alice@example.com for info"}]},
        ]
        result, count = redact_all_sessions(sessions, custom_strings=["alice@example.com"])
        # The email should be redacted
        assert count > 0
