import pytest
from unittest.mock import patch
import sys
from io import StringIO

from src.hfest.cli import main
from src.hfest.version import __version__


class TestMain:
    """Tests for the main entry point function"""

    def test_no_args_shows_help(self):
        """Test that calling with no arguments shows help and returns 1"""
        with patch('sys.argv', ['hfest']):
            with patch('sys.stderr', new=StringIO()) as fake_stderr:
                with pytest.raises(SystemExit) as excinfo:
                    main()
                assert excinfo.value.code == 1
                assert "usage:" in fake_stderr.getvalue()
                assert "Commands" in fake_stderr.getvalue()

    def test_version_flag(self):
        """Test that --version flag shows version and exits"""
        with patch('sys.argv', ['hfest', '--version']):
            with patch('sys.stdout', new=StringIO()) as fake_stdout:
                with pytest.raises(SystemExit) as excinfo:
                    main()  # Call the main function directly
                assert excinfo.value.code == 0
                assert __version__ in fake_stdout.getvalue()

    @patch('src.hfest.commands.estimate_size.handle')
    def test_estimate_size_command(self, mock_handle):
        """Test that estimate-size command routes to the correct handler"""
        mock_handle.return_value = 0
        with patch('sys.argv', ['hfest', 'estimate-size', 'deepseek-ai/DeepSeek-V3']):
            assert main() == 0  # Call the main function directly
            mock_handle.assert_called_once()
            # Verify args passed to handler have the command and arguments
            args = mock_handle.call_args[0][0]
            assert args.command == 'estimate-size'
            assert args.model_id == 'deepseek-ai/DeepSeek-V3'

    @patch('src.hfest.commands.estimate_resource.handle')
    def test_estimate_resource_command(self, mock_handle):
        """Test that estimate-resource command routes to the correct handler"""
        mock_handle.return_value = 0
        with patch('sys.argv', ['hfest', 'estimate-resource', 'deepseek-ai/DeepSeek-V3']):
            assert main() == 0  # Call the main function directly
            mock_handle.assert_called_once()
            args = mock_handle.call_args[0][0]
            assert args.command == 'estimate-resource'
            assert args.model_id == 'deepseek-ai/DeepSeek-V3'

    @patch('src.hfest.commands.config.handle')
    def test_config_command(self, mock_handle):
        """Test that config command routes to the correct handler"""
        mock_handle.return_value = 1
        with patch('sys.argv', ['hfest', 'config', 'list']):
            assert main() == 1  # Call the main function directly
            mock_handle.assert_called_once()
            args = mock_handle.call_args[0][0]
            assert args.command == 'config'
    
    def test_parser_error_handling(self):
        """Test that parser errors are handled correctly"""
        # Test with invalid argument
        with patch('sys.argv', ['hfest', 'estimate-size', '--invalid-arg']):
            with pytest.raises(SystemExit) as excinfo:
                main()  # Call the main function directly
            # Should exit with error code
            assert excinfo.value.code != 0

    @patch('argparse.ArgumentParser.parse_args')
    def test_argument_parser_exception(self, mock_parse_args):
        """Test that SystemExit from argparse is re-raised"""
        mock_parse_args.side_effect = SystemExit(2)
        with patch('sys.argv', ['hfest', '--unknown']):
            with pytest.raises(SystemExit) as excinfo:
                main()  # Call the main function directly
            assert excinfo.value.code == 2
