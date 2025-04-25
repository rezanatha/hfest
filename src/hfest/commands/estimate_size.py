from ..utils.config import read_config
from huggingface_hub import hf_hub_download, scan_cache_dir, HfApi, login
import json
import requests
import re
import sys

def setup_parser(subparsers):
    parser = subparsers.add_parser("estimate-size", help="Estimate model size")
    parser.add_argument("model_id", help="Hugging Face model ID (e.g., meta-llama/Llama-2-7b)")
    return parser


def validate_model_id(model_id):
    pattern = r'^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$'
    return bool(re.match(pattern, model_id))

def estimate_model_files(args):

    config = read_config()

    if not validate_model_id(args.model_id):
        print(f"Invalid model ID format: {args.model_id}")
        print("ERROR: Model ID should follow the format: username/model-name")
        return None

    if config['api_key'] is None:
        print("ERROR: No HuggingFace API key specified.")
        return None
    
    # Initialize the API
    api = HfApi()
    login(config['api_key'])
    sys.stdout.write("Repository Size: calculating...\r")
    sys.stdout.flush()
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
        model_params_size = content.get('safetensors',{}).get('total',0)
        total_used_storage = float(content.get('usedStorage', 0)) / (1024 ** 3)
    elif response.status_code == 401:
        print("ERROR: Authentication error: Invalid or expired API token.")
        return None
    elif response.status_code == 403:
        print("ERROR: Authorization error: You don't have access to this model repository.")
        return None
    elif response.status_code == 404:
        print(f"ERROR: Model not found: {args.model_id} doesn't exist on HuggingFace Hub.")
        return None
    elif response.status_code == 429:
        print("ERROR: Rate limit exceeded. Please try again later.")
        return None
    else:
        print(f"ERROR: API request failed with status code: {response.status_code}")
        if response.content:
            try:
                error_content = json.loads(response.content)
                if 'error' in error_content:
                    print(f"Error message: {error_content['error']}")
            except json.JSONDecodeError:
                print(f"Response content: {response.content.decode('utf-8', errors='replace')}")
        return None
    

    sys.stdout.write("\r" + " " * 50 + "\r") 
    sys.stdout.write(f"Repository Size: {total_used_storage:.2f} GB\n")
    sys.stdout.flush()

    formatted_param_count = f"{model_params_size:,}"
    sys.stdout.write(f"Model Parameter Count: {formatted_param_count}\n")
    sys.stdout.flush()

    model_extensions = (('safetensors',['safetensors']), 
                        ('pytorch', ['bin', 'pt', 'pth']),
                        ('onnx', ['onnx']),
                        )
    
    model_files = {k[0]: [] for k in model_extensions}
    
    if isinstance(repo_files, list) and len(repo_files) > 0:
        for r in repo_files:
            for k, v in model_extensions:
                if r['rfilename'].split('.')[-1] in v:
                    model_files[k].append(r['rfilename'])
    else:
        print("Is an empty repository")
        return None
            
    sys.stdout.write("Estimated Model File Distribution: calculating...\r")
    sys.stdout.flush()

    # for k,v in model_files.items():
    #     print(f"{len(v)} {k}",end=" ")
    # print("\n")

    # fetch real model size 
    estimated_total = {k[0]: 0 for k in model_extensions}
    for i, (model_type, model_name) in enumerate(model_files.items()):
        file_infos = []
        
        for file in model_name[:10]:  # Limit to first 10 files to avoid API abuse
            file_info = api.get_paths_info(repo_id=args.model_id, paths=[file])[0]
            file_infos.append((file, file_info.size if hasattr(file_info, 'size') and file_info.size else "Unknown"))
        # for file, size in file_infos:
        #     if size != "Unknown":
        #         size_mb = size / (1024 * 1024)
        #         print(f"{model_type} File: {file}, Size: {size_mb:.2f} MB")
        #     else:
        #         print(f"{model_type} File: {file}, Size: Unknown")
        
        if any(size != "Unknown" for _, size in file_infos):
            if i == 0:
                sys.stdout.write("\r" + " " * 50 + "\r")
                sys.stdout.write("Estimated Model File Distribution:\n") 
                sys.stdout.flush()

            known_sizes = [size for _, size in file_infos if size != "Unknown"]
            if known_sizes:
                avg_size = sum(known_sizes) / len(known_sizes)
                estimated_total[model_type] = avg_size * len(model_name)
                print(f"  • {model_type}: {len(model_name)} file(s) ({estimated_total[model_type] / (1024**3):.2f} GB)")
        else:
            print(f"  • {model_type}: {len(model_name)} file(s) (0 GB)")

    return estimated_total

def handle(args):
    print(f"Model: {args.model_id}")
    print("----------------------------------------")
    estimated_total = estimate_model_files(args)
    if estimated_total is None:
        return 1
    
    return 0