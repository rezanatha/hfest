from .estimate_size import estimate_model_files
import subprocess

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def get_gpu_info():
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

def compare_single_setup(estimated_total, gpu_info, margin_of_safety = 0.2):
    # how many resources would it take?
    size = estimated_total['safetensors'] / (1024 ** 3)
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
    gpu_info = get_gpu_info()
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
        print("[SINGLE GPU] Model File Size vs Free GPU Memory:")
        compare_single_setup(estimated_total, gpu_info)
        # IF QUANTIZED 
        # IF SHARDED AND DISTRIBUTED
        # if running on a single/distributed system

    # 2. PYTORCH BIN
    elif 'pytorch' in estimated_total:
        print("Comparing model size and free GPU memory on single GPU settings")
        compare_single_setup(estimated_total, gpu_info)
    # 3. ONNX
    elif 'onnx' in estimated_total:
        print("Comparing model size and free GPU memory on single GPU settings")
        compare_single_setup(estimated_total, gpu_info)
    # 4. OTHERS

    
    return 0   