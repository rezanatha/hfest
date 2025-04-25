import pytest
from unittest.mock import patch, MagicMock
import json
import argparse
import sys
import os
from io import StringIO

from src.hfest.commands.estimate_size import setup_parser, validate_model_id, estimate_model_files, handle

# Fixtures
@pytest.fixture
def est_parser():
    """Create and return an estimate-size parser for testing."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    return setup_parser(subparsers)

@pytest.fixture
def valid_model_id():
    """Return a valid model ID."""
    return "deepseek-ai/DeepSeek-R1"

@pytest.fixture
def invalid_model_id():
    """Return an invalid model ID."""
    return "invalid model id with spaces"

@pytest.fixture
def unauthorized_model_id():
    """Return a valid model ID."""
    return "meta-llama/Llama-2-7b"

@pytest.fixture
def mock_config():
    """Mock configuration with API key."""
    return {"api_key": "test_api_key"}

@pytest.fixture
def mock_config_no_key():
    """Mock configuration without API key."""
    return {"api_key": None}

@pytest.fixture
def captured_stderr(monkeypatch):
    """Capture stderr for testing console output."""
    buffer = StringIO()
    monkeypatch.setattr(sys, 'stderr', buffer)
    return buffer

# Tests for validate_model_id function
@pytest.mark.parametrize("model_id, expected", [
    ("meta-llama/Llama-2-7b", True),
    ("deepseek-ai/DeepSeek-V3", True),
    ("microsoft/bitnet-b1.58-2B-4T", True), 
    ("sentence-transformers/all-MiniLM-L6-v2", True),
    ("huggingface/bert-base", True),
    ("", False),
    ("model-without-owner", False),
    ("/owner-without-model", False),
    ("model with spaces/name", False),
    ("owner/model/extra", False),
])
def test_validate_model_id(model_id, expected):
    """Test the validate_model_id function with various inputs."""
    assert validate_model_id(model_id) == expected


# Tests for estimate_model_files function
@patch("src.hfest.commands.estimate_size.read_config")
def test_invalid_model_id_in_estimate(mock_read_config, invalid_model_id, capsys):
    """Test that estimate_model_files rejects invalid model IDs."""
    # Set a return value for read_config
    mock_read_config.return_value = {"api_key": "fake_api_key"}
    
    args = argparse.Namespace(model_id=invalid_model_id)
    result = estimate_model_files(args)
    
    captured = capsys.readouterr()
    stdout_content = captured.out
    
    assert result is None
    assert "Invalid model ID format:" in stdout_content, f"Actual stdout: '{stdout_content}'"


@patch("src.hfest.commands.estimate_size.read_config")
def test_no_api_key(mock_read_config, valid_model_id, mock_config_no_key, capsys):
    """Test behavior when no API key is provided."""
    mock_read_config.return_value = mock_config_no_key
    args = argparse.Namespace(model_id=valid_model_id)
    
    result = estimate_model_files(args)

    captured = capsys.readouterr()
    stdout_content = captured.out
    
    assert result is None
    assert "ERROR: No HuggingFace API key specified" in stdout_content, f"Actual stdout: '{stdout_content}'"


@patch("src.hfest.commands.estimate_size.read_config")
@patch("src.hfest.commands.estimate_size.login")
@patch("src.hfest.commands.estimate_size.HfApi")
@patch("src.hfest.commands.estimate_size.requests.get")
def test_http_status_codes(mock_get, mock_hfapi, mock_login, mock_read_config, valid_model_id, 
                          mock_config, capsys):
    """Test handling of different HTTP status codes."""
    mock_read_config.return_value = mock_config
    mock_api = MagicMock()
    mock_hfapi.return_value = mock_api
    
    # Test different status codes
    status_codes = {
        401: "ERROR: Authentication error",
        403: "ERROR: Authorization error",
        404: "ERROR: Model not found",
        429: "ERROR: Rate limit exceeded",
        500: "ERROR: API request failed with status code: 500"
    }
    
    for status_code, expected_message in status_codes.items():
        # Reset the captured output
        capsys.readouterr()
        
        # Mock response with given status code
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.content = b"{}" 
        mock_get.return_value = mock_response
        
        # Call the function
        args = argparse.Namespace(model_id=valid_model_id)
        result = estimate_model_files(args)

        # Get captured output after function call
        captured = capsys.readouterr()
        stdout_content = captured.out
        
        # Check results
        assert result is None
        assert expected_message in stdout_content, f"Expected '{expected_message}' not found in: '{stdout_content}'"


@patch("src.hfest.commands.estimate_size.read_config")
@patch("src.hfest.commands.estimate_size.login")
@patch("src.hfest.commands.estimate_size.HfApi")
@patch("src.hfest.commands.estimate_size.requests.get")
def test_empty_repository(mock_get, mock_hfapi, mock_login, mock_read_config, valid_model_id,
                         mock_config, capsys):
    """Test behavior with an empty repository."""
    mock_read_config.return_value = mock_config
    mock_api = MagicMock()
    mock_hfapi.return_value = mock_api
    
    # Mock response with empty repository
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = json.dumps({
        "usedStorage": "0",
        "safetensors": {},
        "siblings": []
    }).encode()
    mock_get.return_value = mock_response
    
    args = argparse.Namespace(model_id=valid_model_id)
    
    result = estimate_model_files(args)

    captured = capsys.readouterr()
    stdout_content = captured.out
    
    assert result is None
    assert "Is an empty repository" in stdout_content, f"Actual stdout: '{stdout_content}'"


@patch("src.hfest.commands.estimate_size.read_config")
@patch("src.hfest.commands.estimate_size.login")
@patch("src.hfest.commands.estimate_size.HfApi")
@patch("src.hfest.commands.estimate_size.requests.get")
def test_successful_estimation(mock_get, mock_hfapi, mock_login, mock_read_config, valid_model_id,
                             mock_config, capsys):
    """Test successful model size estimation."""
    mock_read_config.return_value = mock_config
    
    # Mock API and file info
    mock_api = MagicMock()
    file_info = MagicMock()
    file_info.size = 1024 * 1024 * 100  # 100 MB
    mock_api.get_paths_info.return_value = [file_info]
    mock_hfapi.return_value = mock_api
    
    # Mock response with repository data
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = json.dumps({
        "usedStorage": str(1024 * 1024 * 1024 * 10),  # 10 GB
        "safetensors": {"total": 70000000},
        "siblings": [
            {"rfilename": "model-00001-of-00002.safetensors"},
            {"rfilename": "model-00002-of-00002.safetensors"},
            {"rfilename": "pytorch_model-00001-of-00002.bin"},
            {"rfilename": "pytorch_model-00002-of-00002.bin"},
            {"rfilename": "config.json"}
        ]
    }).encode()
    mock_get.return_value = mock_response
    
    args = argparse.Namespace(model_id=valid_model_id)
    
    result = estimate_model_files(args)

    captured = capsys.readouterr()
    stdout_content = captured.out
    
    assert result is not None
    assert "safetensors" in result
    assert "pytorch" in result
    assert "Repository Size: 10.00 GB" in stdout_content
    assert "Model Parameter Count: 70,000,000" in stdout_content
    assert "Estimated Model File Distribution:" in stdout_content


@patch("src.hfest.commands.estimate_size.estimate_model_files")
def test_handle_function_success(mock_estimate, valid_model_id, capsys):
    """Test the handle function with successful estimation."""
    mock_estimate.return_value = {"safetensors": 1000, "pytorch": 2000}
    
    args = argparse.Namespace(model_id=valid_model_id)
    
    result = handle(args)

    captured = capsys.readouterr()
    stdout_content = captured.out
    
    assert result == 0
    assert f"Model: {valid_model_id}" in stdout_content
    mock_estimate.assert_called_once_with(args)


@patch("src.hfest.commands.estimate_size.estimate_model_files")
def test_handle_function_failure(mock_estimate, valid_model_id, capsys):
    """Test the handle function with failed estimation."""
    mock_estimate.return_value = None
    
    args = argparse.Namespace(model_id=valid_model_id)
    
    result = handle(args)

    captured = capsys.readouterr()
    stdout_content = captured.out
    
    assert result == 1
    assert f"Model: {valid_model_id}" in stdout_content
    mock_estimate.assert_called_once_with(args)


def test_setup_parser(est_parser):
    """Test the setup_parser function."""
    args = est_parser.parse_args(["meta-llama/Llama-2-7b"])
    
    assert args.model_id == "meta-llama/Llama-2-7b"