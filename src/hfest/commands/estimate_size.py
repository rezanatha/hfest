from ..utils.config import read_config
from huggingface_hub import hf_hub_download, scan_cache_dir, HfApi, login
import json
import requests
import re

def setup_parser(subparsers):
    parser = subparsers.add_parser("estimate-size", help="Estimate model size")
    parser.add_argument("model_id", help="Hugging Face model ID (e.g., meta-llama/Llama-2-7b)")
    return parser

# TODO:
# validate the input from args.model_id: it should only process model ID like account/model_name
def _validate_model_id(model_id):
    pattern = r'^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$'
    return bool(re.match(pattern, model_id))

def _estimate_model_files(args):

    config = read_config()

    if not _validate_model_id(args.model_id):
        print(f"Invalid model ID format: {args.model_id}")
        print("Model ID should follow the format: username/model-name")
        return 1

    if config['api_key'] is None:
        print("No HuggingFace API key specified.")
        return 1
    
    # Initialize the API
    api = HfApi()
    login(config['api_key'])

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
    
    if isinstance(repo_files, list):
        for r in repo_files:
            for k, v in model_extensions:
                if r['rfilename'].split('.')[-1] in v:
                    model_files[k].append(r['rfilename'])
    else:
        print("is an empty repository")
        return None
            
    print("Model files: ")
    for k,v in model_files.items():
        print(f"{len(v)} {k}",end=" ")
    print("\n")

    # fetch real model size 
    estimated_total = {k[0]: [] for k in model_extensions}
    for model_type, model_name in model_files.items():
        file_infos = []
        for file in model_name[:5]:  # Limit to first 5 files to avoid API abuse
            file_info = api.get_paths_info(repo_id=args.model_id, paths=[file])[0]
            file_infos.append((file, file_info.size if hasattr(file_info, 'size') and file_info.size else "Unknown"))

        # for file, size in file_infos:
        #     if size != "Unknown":
        #         size_mb = size / (1024 * 1024)
        #         print(f"{model_type} File: {file}, Size: {size_mb:.2f} MB")
        #     else:
        #         print(f"{model_type} File: {file}, Size: Unknown")
        
        # For models with many shards, we can estimate total size in GB based on a sample of 5 models
        if any(size != "Unknown" for _, size in file_infos):
            known_sizes = [size for _, size in file_infos if size != "Unknown"]
            if known_sizes:
                avg_size = sum(known_sizes) / len(known_sizes)
                estimated_total[model_type] = avg_size * len(model_name)
                print(f"Estimated used storage for {model_type}: {estimated_total[model_type] / (1024**3):.2f} GB")
            else:
                print("Model size unknown")

    return estimated_total

def handle(args):
    print(f"Estimating size for model: {args.model_id}")
    estimated_total = _estimate_model_files(args)
    
    return 0