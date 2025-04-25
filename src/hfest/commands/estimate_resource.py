from .estimate_size import estimate_model_files
import subprocess
import platform

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

PRECISIONS = {"fp32": 1, "fp16": 2, "int8":4, "int4":8}

def detect_os():
    os_name = platform.system()
    print(f"Operating System: {os_name}")
    if platform.system() == "Windows":
        version = platform.win32_ver()
        print(f"Windows version: {version}")
        return "Windows"
    elif platform.system() == "Darwin":  # macOS
        version = platform.mac_ver()
        print(f"macOS version: {version[0]}")
        return "Darwin"
    elif platform.system() == "Linux":
        try:
            version = None
            if hasattr(platform, 'freedesktop_os_release'):
                version = platform.freedesktop_os_release()['PRETTY_NAME']
            else:
                # For older Python without freedesktop_os_release
                with open('/etc/os-release') as f:
                    os_release = {}
                    for line in f:
                        if '=' in line:
                            k, v = line.rstrip().split('=', 1)
                            os_release[k] = v.strip('"\'')
                version = os_release.get('PRETTY_NAME', 'Unknown')
            print(f"Linux version: {version}")
        except:
            version = "Unknown"
            print(f"Linux version: {version}")
        return "Linux"
    
    print("Operating system is not supported yet")
    return None

def detect_gpu(detected_os):
    if detected_os == "Windows":
        gpu_set = set()
        try:
            cmd = ["wmic","path", "win32_VideoController", "get", "Caption," "AdapterRAM,", "DriverVersion"]
            output = subprocess.check_output(cmd, universal_newlines=True)
            for i, line in enumerate(output.split('\n')):
                if i == 0: # header
                    continue

                if "NVIDIA" in line.upper():
                    gpu_set.add("NVIDIA")
                elif "AMD" or "RADEON" or "ATI" in gpu.upper():
                    gpu_set.add("AMD")
                elif "INTEL" in gpu.upper():
                    gpu_set.add("INTEL")
            return gpu_set
        except:
            return gpu_set
    elif detected_os == "Darwin":
        gpu_set = set()
        try:
            cmd = ["system_profiler", "SPDisplaysDataType"]
            output = subprocess.check_output(cmd, universal_newlines=True)
            for line in output.split('\n'):
                line = line.strip()
                if "Chipset Model: Apple" in line:
                    gpu_set.add("APPLE")
                    return gpu_set
        except:
            return gpu_set
    elif detected_os == "Linux":
        lspci_process = subprocess.Popen(["lspci"], stdout=subprocess.PIPE, text=True)
        grep_process = subprocess.Popen(
            ["grep", "-E", "VGA|Display|3D|Graphics"], 
            stdin=lspci_process.stdout, 
            stdout=subprocess.PIPE, 
            text=True
        )
        lspci_process.stdout.close()  # Allow lspci_process to receive SIGPIPE
        output = grep_process.communicate()[0]

        gpu_info = output.strip().split('\n')
        gpu_set = set()
        for gpu in gpu_info:
            if "NVIDIA" in gpu.upper():
                gpu_set.add("NVIDIA")
            elif "AMD" or "RADEON" or "ATI" in gpu.upper():
                gpu_set.add("AMD")
            elif "INTEL" in gpu.upper():
                gpu_set.add("INTEL")
        return gpu_set

    print("GPU is not detected.")
    return None

def get_nvidia_gpu_info():
    try:
        # Run nvidia-smi command
        output = subprocess.check_output(['nvidia-smi', 
                                          '--query-gpu=index,name,memory.total,memory.used,memory.free', 
                                          '--format=csv,noheader'], 
                                        universal_newlines=True)
        
        # Process the output
        gpu_info = []
        for line in output.strip().split('\n'):
            values = [x.strip() for x in line.split(',')]
            gpu_info.append({
                'index': values[0],
                'name': values[1],
                'memory.total': values[2],
                'memory.used': values[3],
                'memory.free': values[4]
            })
            
        return gpu_info
    
    except subprocess.CalledProcessError as e:
        error_message = e.stderr if hasattr(e, 'stderr') and e.stderr else str(e)
        if "NVIDIA-SMI has failed" in error_message:
            return "NVIDIA driver issues detected. Try updating your drivers."
        elif "not found" in error_message.lower():
            return "nvidia-smi command not found. NVIDIA drivers may not be installed."
        return f"Error running nvidia-smi: {error_message}"
        
    except FileNotFoundError:
        return "nvidia-smi command not found. NVIDIA drivers may not be installed."
        
    except Exception as e:
        return f"Error getting NVIDIA GPU info: {str(e)}"
    
def get_amd_gpu_info():
    try:
        # Run rocm-smi command to get memory info
        output = subprocess.check_output(['rocm-smi', 
                                          '--showmeminfo', 'vram',
                                          '-f', 'csv'], 
                                        universal_newlines=True)
        
        # Get device names in a separate command
        names_output = subprocess.check_output(['rocm-smi', 
                                               '--showname',
                                               '-f', 'csv'], 
                                             universal_newlines=True)
        
        # Process the memory output
        lines = output.strip().split('\n')
        header = [x.strip() for x in lines[0].split(',')]
        
        # Create index to column mapping
        idx_map = {}
        for i, col in enumerate(header):
            if 'GPU ID' in col:
                idx_map['index'] = i
            elif 'Total VRAM' in col:
                idx_map['total'] = i
            elif 'Used VRAM' in col:
                idx_map['used'] = i
        
        # Process device names
        name_lines = names_output.strip().split('\n')
        name_header = [x.strip() for x in name_lines[0].split(',')]
        name_idx_map = {}
        for i, col in enumerate(name_header):
            if 'GPU ID' in col:
                name_idx_map['index'] = i
            elif 'Device Name' in col:
                name_idx_map['name'] = i
        
        # Map GPU IDs to names
        gpu_names = {}
        for i in range(1, len(name_lines)):
            values = [x.strip() for x in name_lines[i].split(',')]
            gpu_id = values[name_idx_map['index']]
            gpu_name = values[name_idx_map['name']]
            gpu_names[gpu_id] = gpu_name
        
        # Process the data and build the result
        gpu_info = []
        for i in range(1, len(lines)):
            values = [x.strip() for x in lines[i].split(',')]
            gpu_id = values[idx_map['index']]
            
            # Calculate free memory
            total_mem = values[idx_map['total']]
            used_mem = values[idx_map['used']]
            
            # Extract numeric values for calculation
            total_num = float(total_mem.split()[0])
            used_num = float(used_mem.split()[0])
            free_num = total_num - used_num
            
            # Get the unit (should be the same for both)
            unit = total_mem.split()[1]
            free_mem = f"{free_num} {unit}"
            
            gpu_info.append({
                'index': gpu_id,
                'name': gpu_names.get(gpu_id, 'Unknown'),
                'memory.total': total_mem,
                'memory.used': used_mem,
                'memory.free': free_mem
            })
            
        return gpu_info
    except FileNotFoundError:
        return "rocm-smi command not found. Make sure ROCm is installed."
    except Exception as e:
        return f"Error getting AMD GPU info: {str(e)}"

def get_intel_gpu_info():
    print("Calculation of Intel GPU memory is not fully supported yet.")
    return []

def get_apple_gpu_info():
    print("Calculation of Apple GPU memory is not fully supported yet.")
    gpu_info = {
                'index': 0,
                'name': None,
                'memory.total': "0 MiB",
                'memory.used': "0 MiB",
                'memory.free': "0 MiB"
            }
    try:
        cmd = ["system_profiler", "SPDisplaysDataType"]
        output = subprocess.check_output(cmd, 
                                    universal_newlines=True)
        model_name = None
        for line in output.split('\n'):
            line = line.strip()
            if "Chipset Model:" in line:
                gpu_info['name'] = line.split("Chipset Model:")[1].strip()
        return [gpu_info]
    except:
        return [gpu_info]

def compare_single_setup(estimated_total, precision, gpu_info, margin_of_safety = 0.2):
    # how many resources would it take?

    size = (estimated_total / PRECISIONS[precision]) / (1024 ** 3)
    for gpu in gpu_info:
        gpu_free = float(gpu['memory.free'].split(" ")[0]) / 1024 
        if size + size * margin_of_safety > gpu_free:
            print(f"  • {RED}[NOT ENOUGH MEMORY]{RESET} on GPU {gpu['index']}: {gpu['name']}. Model size {size:.2f} GB vs Free GPU memory {gpu_free:.2f} GB")
        else:
            print(f"  • {GREEN}[MEMORY CHECK PASSED]{RESET} for {gpu['index']}: {gpu['name']}. Model size {size:.2f} GB vs Free GPU memory {gpu_free:.2f} GB")

def compare_distributed(estimated_total, precision, gpu_info, margin_of_safety = 0.2):
    return 0

def make_recommendation(result):
    return 0

def setup_parser(subparsers):
    parser = subparsers.add_parser("estimate-resource", help = "Estimate model size and resource needed to run the model")
    parser.add_argument("model_id", help="Hugging Face model ID (e.g., meta-llama/Llama-2-7b)")
    parser.add_argument("--filetype", type=str, default="auto", help="Specify model file type for estimation (auto, safetensors, pytorch, onnx)")
    parser.add_argument("--gpu_config", type=str, default="all", help="GPU config the model is running on (all, single, distributed)")
    parser.add_argument("--precision", type=str, default="all", help="precision level of post-training quantization (all, fp32, fp16, int8, int4)")
    return parser

def handle(args):
    print(f"Model: {args.model_id}")
    print("----------------------------------------")
    # estimate model size
    estimated_total = estimate_model_files(args)
    print("----------------------------------------")
    # detect host GPU specifications (from something like nvidia-smi)
    # detect nvidia GPU
    detected_os = detect_os()
    gpu_set = detect_gpu(detected_os)
    gpu_info = []
    if "NVIDIA" in gpu_set:
        gpu_info.extend(get_nvidia_gpu_info())
    if "AMD" in gpu_set:
        gpu_info.extend(get_amd_gpu_info())
    if "INTEL" in gpu_set:
        gpu_info.extend(get_intel_gpu_info())  
    if "APPLE" in gpu_set:
        gpu_info.extend(get_apple_gpu_info())  

    if isinstance(gpu_info, list):
        print(f"Number of Available GPUs: {len(gpu_info)}")
        for gpu in gpu_info:
            free = int(gpu['memory.free'].split(" ")[0])
            total = int(gpu['memory.total'].split(" ")[0])
            print(f"  • GPU {gpu['index']}: {gpu['name']} ({total - free}/{total} MB)")
            # print(f"  • Total memory: {gpu['memory.total']}")
            # print(f"  • Used memory: {gpu['memory.used']}")
            # print(f"  • Free memory: {gpu['memory.free']}")
    else:
        print(gpu_info)

    # compare gpu spec with model size, is it possible to run on it?

    # MODEL PRIORITY
    # 1. SAFETENSORS
    print("----------------------------------------")
    precision_levels = ['fp32','fp16', 'int8','int4']
    if estimated_total.get('safetensors', 0) > 0:
        for q in precision_levels:
            print(f"[{q.upper()} - SINGLE] Safetensors Model File Size vs Free GPU Memory:")
            # Single Settings, all models are fitted into GPU
            compare_single_setup(estimated_total['safetensors'], q, gpu_info)
            # IF SHARDED AND DISTRIBUTED
            compare_distributed(estimated_total['safetensors'], q, gpu_info)

    # 2. PYTORCH BIN
    elif estimated_total.get('pytorch', 0) > 0:
        for q in precision_levels:
            print(f"[{q.upper()} - SINGLE] PyTorch Model File Size vs Free GPU Memory:")
            # Single Settings, all models are fitted into GPU
            compare_single_setup(estimated_total['pytorch'], q, gpu_info)
            # IF SHARDED AND DISTRIBUTED
            compare_distributed(estimated_total['pytorch'], q, gpu_info)
    # 3. ONNX ()
    elif estimated_total.get('onnx', 0) > 0:
        for q in precision_levels:
            print(f"[{q.upper()} - SINGLE] ONNX Model File Size vs Free GPU Memory:")
            # Single Settings, all models are fitted into GPU
            compare_single_setup(estimated_total['onnx'], q, gpu_info)
            # IF SHARDED AND DISTRIBUTED
            compare_distributed(estimated_total['onnx'], q, gpu_info)
    # 4. OTHERS

    
    return 0   