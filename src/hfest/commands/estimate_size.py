from utils.config import read_config
from huggingface_hub import hf_hub_download, scan_cache_dir, HfApi, login

def setup_parser(subparsers):
    parser = subparsers.add_parser("estimate-size", help="Estimate model size")
    parser.add_argument("model_id", help="Hugging Face model ID (e.g., meta-llama/Llama-2-7b)")
    return parser

def handle(args):
    print(f"Estimating size for model: {args.model_id}")
    config = read_config()

    if config['api_key'] is None:
        print("No HuggingFace API key specified.")
        return 1
    
    # Initialize the API
    api = HfApi()
    login(config['api_key'])

    # connect to model ID
    model_id = args.model_id
    model_files = api.list_repo_files(model_id)
    
    model_extensions = ('.safetensors', '.bin', '.pt', '.ckpt')
    model_files = [f for f in model_files if any(f.endswith(ext) for ext in model_extensions)]
    
    # fetch model size
    file_infos = []
    for file in model_files[:10]:  # Limit to first 5 files to avoid API abuse
        file_info = api.get_paths_info(repo_id=model_id, paths=[file])[0]
        file_infos.append((file, file_info.size if hasattr(file_info, 'size') and file_info.size else "Unknown"))

    for file, size in file_infos:
        if size != "Unknown":
            size_mb = size / (1024 * 1024)
            print(f"File: {file}, Size: {size_mb:.2f} MB")
        else:
            print(f"File: {file}, Size: Unknown")
        
    # For models with many shards, we can estimate total size based on a sample
    # estimate model size
    if len(model_files) > 5 and any(size != "Unknown" for _, size in file_infos):
        known_sizes = [size for _, size in file_infos if size != "Unknown"]
        if known_sizes:
            avg_size = sum(known_sizes) / len(known_sizes)
            estimated_total = avg_size * len(model_files)
            print(f"\nEstimated total size (based on sample): {estimated_total / (1024**3):.2f} GB")
            print(f"Total number of model files: {len(model_files)}")
    return 0