from huggingface_hub import hf_hub_download, scan_cache_dir, HfApi, login
from dotenv import load_dotenv
import os
# Initialize the API
api = HfApi()

# loat .env
load_dotenv()

# Login with your token
login(os.getenv("HF_API_KEY"))

model_id = "microsoft/bitnet-b1.58-2B-4T"
model_files = api.list_repo_files(model_id)

model_extensions = ('.safetensors', '.bin', '.pt', '.ckpt')
model_files = [f for f in model_files if any(f.endswith(ext) for ext in model_extensions)]
print(f"got {len(model_files)} model files")

file_infos = []
for file in model_files[:10]:  # Limit to first 5 files to avoid API abuse
    file_info = api.get_paths_info(repo_id=model_id, paths=[file])[0]
    file_infos.append((file, file_info.size if hasattr(file_info, 'size') and file_info.size else "Unknown"))

#print(file_infos)

for file, size in file_infos:
    if size != "Unknown":
        size_mb = size / (1024 * 1024)
        print(f"File: {file}, Size: {size_mb:.2f} MB")
    else:
        print(f"File: {file}, Size: Unknown")
        
# For models with many shards, we can estimate total size based on a sample
if len(model_files) > 5 and any(size != "Unknown" for _, size in file_infos):
    known_sizes = [size for _, size in file_infos if size != "Unknown"]
    if known_sizes:
        avg_size = sum(known_sizes) / len(known_sizes)
        estimated_total = avg_size * len(model_files)
        print(f"\nEstimated total size (based on sample): {estimated_total / (1024**3):.2f} GB")
        print(f"Total number of model files: {len(model_files)}")
    else:
        print("Model size unknown")
