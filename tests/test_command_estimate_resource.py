import pytest
from unittest.mock import patch, mock_open, Mock
import subprocess
import io
import sys

from src.hfest.commands.estimate_resource import detect_os, detect_gpu, get_nvidia_gpu_info, get_intel_gpu_info, get_amd_gpu_info, get_apple_gpu_info, compare_single_setup, handle


# detect os
# detect OS windows
# detect OS darwin
# detect OS linux
# can't detect OS
class TestDetectOS:
    
    @patch('platform.system')
    @patch('platform.win32_ver')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_windows_detection(self, mock_stdout, mock_win32_ver, mock_system):
        # Mock the platform.system() to return "Windows"
        mock_system.return_value = "Windows"
        # Mock the platform.win32_ver() to return a sample version
        mock_win32_ver.return_value = ('10', '10.0.19042', '', 'Multiprocessor Free')
        
        # Call the function
        result = detect_os()
        
        # Check the return value
        assert result == "Windows"
        # Check the printed output
        expected_output = "Operating System: Windows\nWindows version: ('10', '10.0.19042', '', 'Multiprocessor Free')\n"
        assert mock_stdout.getvalue() == expected_output
        
    @patch('platform.system')
    @patch('platform.mac_ver')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_macos_detection(self, mock_stdout, mock_mac_ver, mock_system):
        # Mock the platform.system() to return "Darwin"
        mock_system.return_value = "Darwin"
        # Mock the platform.mac_ver() to return a sample version
        mock_mac_ver.return_value = ('12.6.0', ('', '', ''), 'x86_64')
        
        # Call the function
        result = detect_os()
        
        # Check the return value
        assert result == "Darwin"
        # Check the printed output
        expected_output = "Operating System: Darwin\nmacOS version: 12.6.0\n"
        assert mock_stdout.getvalue() == expected_output
        
    @patch('platform.system')
    @patch('platform.freedesktop_os_release', create=True)
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_linux_detection_with_freedesktop(self, mock_stdout, mock_freedesktop, mock_system):
        # Mock the platform.system() to return "Linux"
        mock_system.return_value = "Linux"
        # Mock platform.freedesktop_os_release to return a sample version
        mock_freedesktop.return_value = {'PRETTY_NAME': 'Ubuntu 22.04.1 LTS'}
        
        # Set hasattr to return True for freedesktop_os_release
        with patch('builtins.hasattr', return_value=True):
            # Call the function
            result = detect_os()
            
            # Check the return value
            assert result == "Linux"
            # Check the printed output
            expected_output = "Operating System: Linux\nLinux version: Ubuntu 22.04.1 LTS\n"
            assert mock_stdout.getvalue() == expected_output
            
    @patch('platform.system')
    @patch('builtins.open', new_callable=mock_open, read_data='PRETTY_NAME="Debian GNU/Linux 11"\nVERSION="11 (bullseye)"\n')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_linux_detection_with_os_release(self, mock_stdout, mock_file, mock_system):
        # Mock the platform.system() to return "Linux"
        mock_system.return_value = "Linux"
        
        # Set hasattr to return False for freedesktop_os_release
        with patch('builtins.hasattr', return_value=False):
            # Call the function
            result = detect_os()
            
            # Check the return value
            assert result == "Linux"
            # Check the printed output
            expected_output = "Operating System: Linux\nLinux version: Debian GNU/Linux 11\n"
            assert mock_stdout.getvalue() == expected_output
            
    @patch('platform.system')
    @patch('builtins.open')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_linux_detection_with_exception(self, mock_stdout, mock_file, mock_system):
        # Mock the platform.system() to return "Linux"
        mock_system.return_value = "Linux"
        # Mock open to raise an exception
        mock_file.side_effect = Exception("File not found")
        
        # Set hasattr to return False for freedesktop_os_release
        with patch('builtins.hasattr', return_value=False):
            # Call the function
            result = detect_os()
            
            # Check the return value
            assert result == "Linux"
            # Check the printed output
            expected_output = "Operating System: Linux\nLinux version: Unknown\n"
            assert mock_stdout.getvalue() == expected_output
            
    @patch('platform.system')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_unsupported_os_detection(self, mock_stdout, mock_system):
        # Mock the platform.system() to return an unsupported OS
        mock_system.return_value = "SomeOtherOS"
        
        # Call the function
        result = detect_os()
        
        # Check the return value
        assert result is None
        # Check the printed output
        expected_output = "Operating System: SomeOtherOS\nOperating system is not supported yet\n"
        assert mock_stdout.getvalue() == expected_output

# detect gpu
# able to detect GPU in windows
# unable to detect GPU in windows
# able detect GPU in darwin
# unable to detect GPU in darwin
# able detect GPU in linux
# unable to detect GPU in linux

class TestDetectGPU:
    
    @patch('subprocess.check_output')
    def test_windows_nvidia_gpu_detection(self, mock_check_output):
        # Mock subprocess.check_output to return Windows NVIDIA GPU info
        mock_output = """Caption  AdapterRAM  DriverVersion
NVIDIA GeForce RTX 3080  8589934592  31.0.15.2225
"""
        mock_check_output.return_value = mock_output
        
        # Call the function with "Windows" OS
        result = detect_gpu("Windows")
        
        # Check mock was called with correct arguments
        mock_check_output.assert_called_with(
            ["wmic", "path", "win32_VideoController", "get", "Caption," "AdapterRAM,", "DriverVersion"],
            universal_newlines=True
        )
        
        # Check the return value
        assert result == {"NVIDIA"}
    
    @patch('subprocess.check_output')
    def test_windows_amd_gpu_detection(self, mock_check_output):
        # Mock subprocess.check_output to return Windows AMD GPU info
        mock_output = """Caption  AdapterRAM  DriverVersion
AMD Radeon RX 6800 XT  16777216000  21.9.1
"""
        mock_check_output.return_value = mock_output
        
        # Call the function with "Windows" OS
        result = detect_gpu("Windows")
        
        # Check the return value
        assert result == {"AMD"}
    
    @patch('subprocess.check_output')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_windows_intel_gpu_detection(self, mock_stdout, mock_check_output):
        # Mock subprocess.check_output to return Windows Intel GPU info
        mock_output = """Caption  AdapterRAM  DriverVersion
Intel(R) UHD Graphics 630  2147483648  27.20.100.9466
"""
        mock_check_output.return_value = mock_output
        
        # Call the function with "Windows" OS
        result = detect_gpu("Windows")
        
        # Check the return value
        assert result == {"INTEL"}
    
    @patch('subprocess.check_output')
    def test_windows_multiple_gpus(self, mock_check_output):
        # Mock subprocess.check_output to return Windows with multiple GPUs
        mock_output = """Caption  AdapterRAM  DriverVersion
NVIDIA GeForce RTX 3080  8589934592  31.0.15.2225
Intel(R) UHD Graphics 630  2147483648  27.20.100.9466
"""
        mock_check_output.return_value = mock_output
        
        # Call the function with "Windows" OS
        result = detect_gpu("Windows")
        
        # Check the return value contains both GPUs
        assert result == {"NVIDIA", "INTEL"}
    
    @patch('subprocess.check_output')
    def test_windows_no_recognized_gpu(self, mock_check_output):
        # Mock subprocess.check_output to return Windows with unrecognized GPU
        mock_output = """Caption  AdapterRAM  DriverVersion
Generic Display Adapter  1073741824  1.0.0.1
"""
        mock_check_output.return_value = mock_output
        
        # Call the function with "Windows" OS
        result = detect_gpu("Windows")
        
        # Check the return value is an empty set
        assert result == set()
    
    @patch('subprocess.check_output')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_windows_exception_handling(self, mock_stdout, mock_check_output):
        # Mock subprocess.check_output to raise an exception
        mock_check_output.side_effect = subprocess.SubprocessError("Command failed")
        
        # Call the function with "Windows" OS
        result = detect_gpu("Windows")
        
        # Check the return value is an empty set
        assert result == set()
    
    @patch('subprocess.check_output')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_macos_apple_gpu_detection(self, mock_stdout, mock_check_output):
        # Mock subprocess.check_output to return macOS Apple GPU info
        mock_output = """Graphics/Displays:
    Apple M1:
      Chipset Model: Apple M1
      Type: GPU
      Bus: Built-In
      Total Number of Cores: 8
"""
        mock_check_output.return_value = mock_output
        
        # Call the function with "Darwin" OS
        result = detect_gpu("Darwin")
        
        # Check mock was called with correct arguments
        mock_check_output.assert_called_with(
            ["system_profiler", "SPDisplaysDataType"],
            universal_newlines=True
        )
        
        # Check the return value
        assert result == {"APPLE"}
    
    @patch('subprocess.check_output')
    def test_macos_no_gpu_detection(self, mock_check_output):
        # Mock subprocess.check_output to return output without Apple GPU info
        mock_output = """Graphics/Displays:
    Intel Iris Pro:
      Chipset Model: Intel Iris Pro
      Type: GPU
      Bus: Built-In
"""
        mock_check_output.return_value = mock_output
        
        # Call the function with "Darwin" OS
        result = detect_gpu("Darwin")
        
        # Check mock was called correctly
        mock_check_output.assert_called_with(
            ["system_profiler", "SPDisplaysDataType"],
            universal_newlines=True
        )
        
        # Check the return value - no Apple GPU detected
        assert result == set()
    
    @patch('subprocess.check_output')
    def test_macos_exception_handling(self, mock_check_output):
        # Mock subprocess.check_output to raise an exception
        mock_check_output.side_effect = subprocess.SubprocessError("Command failed")
        
        # Call the function with "Darwin" OS
        result = detect_gpu("Darwin")
        
        # Check the return value is an empty set
        assert result == set()
    
    @patch('subprocess.Popen')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_linux_nvidia_gpu_detection(self, mock_stdout, mock_popen):
        # Create mock objects for the processes
        mock_lspci_process = Mock()
        mock_grep_process = Mock()
        
        # Configure the behavior of the mocks
        mock_popen.side_effect = [mock_lspci_process, mock_grep_process]
        mock_lspci_process.stdout = Mock()
        mock_grep_process.communicate.return_value = ["""01:00.0 3D controller: NVIDIA Corporation GP108M [GeForce MX150] (rev a1)
""", None]
        
        # Call the function with "Linux" OS
        result = detect_gpu("Linux")
        
        # Check the return value
        assert result == {"NVIDIA"}
    
    @patch('subprocess.Popen')
    def test_linux_amd_gpu_detection(self,
                                     mock_popen):
        # Create mock objects for the processes
        mock_lspci_process = Mock()
        mock_grep_process = Mock()
        
        # Configure the behavior of the mocks
        mock_popen.side_effect = [mock_lspci_process, mock_grep_process]
        mock_lspci_process.stdout = Mock()
        mock_grep_process.communicate.return_value = ["""01:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 21 [Radeon RX 6800/6800 XT / 6900 XT] (rev c1)
""", None]
        
        # Call the function with "Linux" OS
        result = detect_gpu("Linux")
        
        # Check the return value
        assert result == {"AMD"}
    
    @patch('subprocess.Popen')
    def test_linux_intel_gpu_detection(self, mock_popen):
        # Create mock objects for the processes
        mock_lspci_process = Mock()
        mock_grep_process = Mock()
        
        # Configure the behavior of the mocks
        mock_popen.side_effect = [mock_lspci_process, mock_grep_process]
        mock_lspci_process.stdout = Mock()
        mock_grep_process.communicate.return_value = ["""00:02.0 VGA compatible controller: Intel Corporation UHD Graphics 620 (rev 07)
""", None]
        
        # Call the function with "Linux" OS
        result = detect_gpu("Linux")
        # Check the return value
        assert result == {"INTEL"}
    
    @patch('subprocess.Popen')
    def test_linux_multiple_gpus(self,
                                 mock_popen):
        # Create mock objects for the processes
        mock_lspci_process = Mock()
        mock_grep_process = Mock()
        
        # Configure the behavior of the mocks
        mock_popen.side_effect = [mock_lspci_process, mock_grep_process]
        mock_lspci_process.stdout = Mock()
        mock_grep_process.communicate.return_value = ["""00:02.0 VGA compatible controller: Intel Corporation UHD Graphics 620 (rev 07)
01:00.0 3D controller: NVIDIA Corporation GP108M [GeForce MX150] (rev a1)
""", None]
        
        # Call the function with "Linux" OS
        result = detect_gpu("Linux")
        # Check the return value includes both Intel and NVIDIA
        assert result == {"INTEL", "NVIDIA"}
    
    @patch('subprocess.Popen')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_linux_no_output(self, mock_stdout, mock_popen):
        # Create mock objects for the processes
        mock_lspci_process = Mock()
        mock_grep_process = Mock()
        
        # Configure the behavior of the mocks
        mock_popen.side_effect = [mock_lspci_process, mock_grep_process]
        mock_lspci_process.stdout = Mock()
        mock_grep_process.communicate.return_value = ["", None]  # Empty output
        
        # Call the function with "Linux" OS
        result = detect_gpu("Linux")
        
        # Check the return value is an empty set
        assert result == set()
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_unsupported_os(self, mock_stdout):
        # Call the function with an unsupported OS
        result = detect_gpu("SomeOtherOS")
        
        # Check the return value
        assert result == set()
        # Check the printed output
        assert "GPU is not detected." in mock_stdout.getvalue()

# estimate-resource
# If model is invalid
# If model is valid
# If model is valid but

if __name__ == "__main__":
    pytest.main()