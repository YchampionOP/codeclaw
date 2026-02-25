"""Tests for codeclaw.classifier."""

from codeclaw.classifier import classify_trajectory


def test_classifies_correction_loop():
    session = {
        "messages": [
            {"role": "assistant", "content": "Try this"},
            {"role": "user", "content": "that's not right, please fix"},
        ]
    }
    assert classify_trajectory(session) == "correction_loop"


def test_classifies_debugging_trace():
    session = {
        "messages": [
            {"role": "assistant", "tool_uses": [{"tool": "bash", "input": "pytest"}]},
            {"role": "assistant", "content": "error: failed test"},
        ]
    }
    assert classify_trajectory(session) == "debugging_trace"


def test_classifies_refactor():
    session = {
        "messages": [
            {"role": "user", "content": "please refactor this file"},
            {"role": "assistant", "content": "ok"},
        ]
    }
    assert classify_trajectory(session) == "refactor"


def test_classifies_iterative_build():
    session = {
        "messages": [{"role": "assistant", "tool_uses": [{"tool": "bash"}]}] * 9,
    }
    assert classify_trajectory(session) == "iterative_build"


def test_classifies_sft_clean_default():
    session = {"messages": [{"role": "user", "content": "hello"}]}
    assert classify_trajectory(session) == "sft_clean"
