from .estimate_size import _estimate_model_files
import subprocess

def get_gpu_info():
    try:
        # Run nvidia-smi command
        output = subprocess.check_output(['nvidia-smi', 
                                          '--query-gpu=index,name,memory.total,memory.used,memory.free', '--format=csv,noheader'], 
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
    size_mb = int(estimated_total['safetensors'] / (1024 * 1024))
    for gpu in gpu_info:
        gpu_mb = int(gpu['memory.free'].split(" ")[0])
        if size_mb + size_mb * margin_of_safety > gpu_mb :
            print(f"Insufficient memory on GPU {gpu['index']}: {gpu['name']}: Model size {size_mb} MB vs Free GPU memory {gpu_mb} MB")
        else:
            print(f"Model would fit on GPU {gpu['index']}: {gpu['name']}: Model size {size_mb} MB vs Free GPU memory {gpu_mb} MB")

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
    print(f"Estimating resources for model: {args.model_id}")
    # estimate model size
    estimated_total = _estimate_model_files(args)
    # detect host GPU specifications (from something like nvidia-smi)
    # detect nvidia GPU
    gpu_info = get_gpu_info()
    if isinstance(gpu_info, list):
        print(f"Number of available GPUs: {len(gpu_info)}")
        for gpu in gpu_info:
            print(f"GPU {gpu['index']}: {gpu['name']}")
            print(f"  Total memory: {gpu['memory.total']}")
            print(f"  Used memory: {gpu['memory.used']}")
            print(f"  Free memory: {gpu['memory.free']}")
    else:
        print(gpu_info)

    # compare gpu spec with model size, is it possible to run on it?
    # MODEL PRIORITY
    # 1. SAFETENSORS
    if 'safetensors' in estimated_total:
        # Original Settings, all model is fitted into GPU
        print("Comparing model size and free GPU memory on single GPU settings")
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