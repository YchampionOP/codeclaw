"""Tests for codeclaw.setup_wizard — interactive setup wizard."""

import pytest

from codeclaw.setup_wizard import _parse_dataset_repo


class TestParseDatasetRepo:
    def test_plain_repo(self):
        assert _parse_dataset_repo("username/cc-logs") == "username/cc-logs"

    def test_full_url(self):
        assert _parse_dataset_repo("https://huggingface.co/datasets/username/cc-logs") == "username/cc-logs"

    def test_full_url_with_trailing_slash(self):
        assert _parse_dataset_repo("https://huggingface.co/datasets/user/repo/") == "user/repo"

    def test_empty(self):
        assert _parse_dataset_repo("") is None

    def test_no_slash(self):
        assert _parse_dataset_repo("noslash") is None

    def test_multiple_slashes(self):
        assert _parse_dataset_repo("a/b/c") is None

    def test_leading_slash(self):
        assert _parse_dataset_repo("/bad") is None

    def test_trailing_slash(self):
        assert _parse_dataset_repo("bad/") is None

    def test_whitespace_stripped(self):
        assert _parse_dataset_repo("  user/repo  ") == "user/repo"

    def test_http_url(self):
        assert _parse_dataset_repo("http://huggingface.co/datasets/user/repo") == "user/repo"
