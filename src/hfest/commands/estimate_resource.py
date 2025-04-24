from .estimate_size import estimate_model_files
import subprocess
import platform

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def detect_os():
    os_name = platform.system()
    print(f"Operating System: {os_name}")
    if platform.system() == "Windows":
        version = platform.win32_ver()
        print(f"Windows version: {version}")
        return "Windows"
    elif platform.system() == "Darwin":  # macOS
        version = platform.mac_ver()
        print(f"macOS version: {version}")
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
        return gpu_set
    elif detected_os == "Darwin":
        gpu_set = set()
        try:
            cmd = ["system_profiler", "SPDisplaysDataType"]
            output = subprocess.check_output(cmd, 
                                        universal_newlines=True)
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
    except:
        return "No NVIDIA GPUs detected or nvidia-smi failed to run"
    
def get_amd_gpu_info():
    return []

def get_intel_gpu_info():
    return []

def get_apple_gpu_info():
    print("Calculation of GPU memory on Apple devices is not supported yet.")
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

def compare_single_setup(estimated_total: list, gpu_info, margin_of_safety = 0.2):
    # how many resources would it take?
    size = estimated_total / (1024 ** 3)
    for gpu in gpu_info:
        gpu_free = float(gpu['memory.free'].split(" ")[0]) / 1024 
        if size + size * margin_of_safety > gpu_free:
            print(f"  • {RED}[NOT ENOUGH MEMORY]{RESET} on GPU {gpu['index']}: {gpu['name']}. Model size {size:.2f} GB vs Free GPU memory {gpu_free:.2f} GB")
        else:
            print(f"  • {GREEN}[MEMORY CHECK PASSED]{RESET} for {gpu['index']}: {gpu['name']}. Model size {size:.2f} GB vs Free GPU memory {gpu_free:.2f} GB")

def compare_quantized(estimated_total, gpu_info):
    return 0

def compare_distributed(estimated_total, gpu_info):
    return 0

def make_recommendation(result):
    return 0

def setup_parser(subparsers):
    parser = subparsers.add_parser("estimate-resource", help = "Estimate model size and resource needed to run the model")
    parser.add_argument("model_id", help="Hugging Face model ID (e.g., meta-llama/Llama-2-7b)")
    parser.add_argument("--gpu_config", type=int, default=0, help="GPU config the model is running on (0: auto, 1: single setup, 2: distributed setup, )")
    parser.add_argument("--quantization", type=int, default=0, help="level of quantization (0: auto, 1: minimal, 2: adequate, 3: aggresive)")
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
        gpu_info.append(get_nvidia_gpu_info())
    if "AMD" in gpu_set:
        gpu_info.append(get_amd_gpu_info())
    if "INTEL" in gpu_set:
        gpu_info.append(get_intel_gpu_info())  
    if "APPLE" in gpu_set:
        gpu_info.append(get_apple_gpu_info())  

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
    if 'safetensors' in estimated_total:
        # Original Settings, all model is fitted into GPU
        print("[SINGLE GPU] Safetensors Model File Size vs Free GPU Memory:")
        compare_single_setup(estimated_total['safetensors'], gpu_info)
        # IF QUANTIZED 
        # IF SHARDED AND DISTRIBUTED
        # if running on a single/distributed system

    # 2. PYTORCH BIN
    elif 'pytorch' in estimated_total:
        print("[SINGLE GPU] PyTorch Model File Size vs Free GPU Memory:")
        compare_single_setup(estimated_total['pytorch'], gpu_info)
    # 3. ONNX ()
    elif 'onnx' in estimated_total:
        print("[SINGLE GPU] Model File Size vs Free GPU Memory:")
        compare_single_setup(estimated_total['onnx'], gpu_info)
    # 4. OTHERS

    
    return 0   