import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from codeclaw.cli.export import export_to_jsonl

def test_export_to_jsonl_open_error(capsys):
    """Test that export_to_jsonl handles OSError when opening the output file."""
    # Mocking necessary arguments for export_to_jsonl
    selected_projects = []
    output_path = Path("fake_path.jsonl")
    anonymizer = MagicMock()

    # Mock load_config to return an empty dict
    with patch("codeclaw.cli.export.load_config", return_value={}):
        # Mock open to raise OSError
        with patch("builtins.open", side_effect=OSError("Permission denied")):
            with pytest.raises(SystemExit) as excinfo:
                export_to_jsonl(
                    selected_projects,
                    output_path,
                    anonymizer
                )

            # Verify exit code
            assert excinfo.value.code == 1

            # Verify error message in stderr
            captured = capsys.readouterr()
            assert f"Error: cannot write to {output_path}: Permission denied" in captured.err
