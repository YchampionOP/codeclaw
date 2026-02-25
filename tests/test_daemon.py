"""Tests for codeclaw.daemon — retry logic and SystemExit propagation."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
import requests

from codeclaw.daemon import RETRY_BACKOFF_SECONDS, _poll_once


@pytest.fixture
def _base_config():
    return {
        "auto_push": True,
        "min_sessions_before_push": 1,
        "repo": "user/dataset",
        "last_synced_at": None,
        "redact_usernames": [],
        "redact_strings": [],
    }


class TestPollOnceRetry:
    def test_system_exit_propagates(self, _base_config):
        """SystemExit from push_to_huggingface must NOT be caught by the retry loop."""
        logger = logging.getLogger("test")

        with (
            patch("codeclaw.daemon.load_config", return_value=dict(_base_config)),
            patch("codeclaw.daemon.save_config"),
            patch("codeclaw.daemon._scan_changed_project_dirs", return_value={"proj"}),
            patch("codeclaw.daemon.discover_projects", return_value=[{"dir_name": "proj"}]),
            patch(
                "codeclaw.daemon.export_to_jsonl",
                return_value={"sessions": 1, "projects": ["proj"]},
            ),
            patch("codeclaw.daemon._append_file"),
            patch("codeclaw.daemon._count_jsonl", return_value=1),
            patch(
                "codeclaw.daemon.push_to_huggingface",
                side_effect=SystemExit(1),
            ),
            patch("codeclaw.daemon.time.sleep"),
        ):
            with pytest.raises(SystemExit):
                _poll_once(logger)

    def test_transient_error_is_retried(self, _base_config):
        """requests.ConnectionError (Exception subclass) must be retried the full backoff sequence."""
        logger = logging.getLogger("test")
        push_mock = MagicMock(side_effect=requests.ConnectionError("network down"))

        with (
            patch("codeclaw.daemon.load_config", return_value=dict(_base_config)),
            patch("codeclaw.daemon.save_config"),
            patch("codeclaw.daemon._scan_changed_project_dirs", return_value={"proj"}),
            patch("codeclaw.daemon.discover_projects", return_value=[{"dir_name": "proj"}]),
            patch(
                "codeclaw.daemon.export_to_jsonl",
                return_value={"sessions": 1, "projects": ["proj"]},
            ),
            patch("codeclaw.daemon._append_file"),
            patch("codeclaw.daemon._count_jsonl", return_value=1),
            patch("codeclaw.daemon.push_to_huggingface", push_mock),
            patch("codeclaw.daemon.time.sleep"),
        ):
            _poll_once(logger)

        assert push_mock.call_count == len(RETRY_BACKOFF_SECONDS) + 1
