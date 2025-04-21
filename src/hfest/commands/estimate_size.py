from utils.config import read_config
from huggingface_hub import hf_hub_download, scan_cache_dir, HfApi, login
import json

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
    import requests
    response = requests.get(
        f"https://huggingface.co/api/models/{args.model_id}",
        params={'fields': ['usedStorage', 'safetensors', 'siblings']},
        headers={"Authorization":f"Bearer {config['api_key']}"}
        )
    total_used_storage = None
    model_params_size = None
    repo_files = None
    if response.status_code == 200:
        content = json.loads(response.content)
        repo_files = content.get('siblings', None)
        model_params_size = content.get('safetensors',{}).get('total',None)
        total_used_storage = float(content.get('usedStorage', 0)) / (1024 ** 3)

    print(f"Total used storage for the whole repository: {total_used_storage:.2f} GB")
    print(f"Parameter size: {model_params_size}")

    model_extensions = (('safetensors',['safetensors']), ('pytorch', ['bin', 'pt', 'pth']))
    model_files = {k[0]: [] for k in model_extensions}
    for r in repo_files:
        for k, v in model_extensions:
            if r['rfilename'].split('.')[-1] in v:
                model_files[k].append(r['rfilename'])
    print("Model files: ")
    for k,v in model_files.items():
        print(f"{len(v)} {k}",end=" ")
    print("\n")

    # # connect to model ID
    # model_id = args.model_id
    # model_files = api.list_repo_files(model_id)

    # model_files = [f for f in model_files if any(f.endswith(ext) for ext in model_extensions)]
    # print(f"got {len(model_files)} model files, only print the first 5")
    
    # fetch real model size 
    for model_type, model_name in model_files.items():
        file_infos = []
        for file in model_name[:5]:  # Limit to first 5 files to avoid API abuse
            file_info = api.get_paths_info(repo_id=args.model_id, paths=[file])[0]
            file_infos.append((file, file_info.size if hasattr(file_info, 'size') and file_info.size else "Unknown"))

        for file, size in file_infos:
            if size != "Unknown":
                size_mb = size / (1024 * 1024)
                print(f"{model_type} File: {file}, Size: {size_mb:.2f} MB")
            else:
                print(f"{model_type} File: {file}, Size: Unknown")
        
        # For models with many shards, we can estimate total size in GB based on a sample of 5 models
        if any(size != "Unknown" for _, size in file_infos):
            known_sizes = [size for _, size in file_infos if size != "Unknown"]
            if known_sizes:
                avg_size = sum(known_sizes) / len(known_sizes)
                estimated_total = avg_size * len(model_name)
                print(f"Estimated used storage for {model_type}: {estimated_total / (1024**3):.2f} GB")
            else:
                print("Model size unknown")

    
    return 0