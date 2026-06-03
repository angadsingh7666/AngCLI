import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

# Define the models you want to download here as (repo_id, "folder_name")
MODELS_TO_DOWNLOAD = [
    
    # Add more models here following the same format: ("repo_id", "new_model_name")
]

def download_model(repo_id: str, model_name: str):
    # Construct the target directory: ../models/new_model_name
    base_dir = Path("../models")
    local_dir = base_dir / model_name
    
    # Ensure the base directory exists
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if the model is already downloaded (Fixed the logical OR bug from the original)
    has_config = (local_dir / "config.json").exists()
    has_safetensors = (local_dir / "model.safetensors").exists() or (local_dir / "model.safetensors.index.json").exists()
    
    if has_config and has_safetensors:
        print(f"✅ Model '{model_name}' already exists at {local_dir}")
        return

    print(f"\n🌐 Downloading '{repo_id}' to '{local_dir}'...")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"✅ Download complete for '{model_name}'. Format: HuggingFace Transformers (.safetensors)")
    except Exception as e:
        print(f"❌ Download failed for '{model_name}': {e}")
        # Note: We do not sys.exit(1) here so the script can continue downloading the remaining models

if __name__ == "__main__":
    # Optional: Allow overriding the list via command line arguments
    # Usage: python scripts/download_model.py repo_id1 folder_name1 repo_id2 folder_name2
    if len(sys.argv) > 1:
        if len(sys.argv) % 2 != 1:
            print("Usage: python scripts/download_model.py <repo_id1> <folder_name1> [<repo_id2> <folder_name2> ...]")
            sys.exit(1)
        
        # Parse command line arguments into pairs
        args = sys.argv[1:]
        models_to_run = [(args[i], args[i+1]) for i in range(0, len(args), 2)]
    else:
        # Use the predefined list if no arguments are provided
        models_to_run = MODELS_TO_DOWNLOAD

    print(f"Starting batch download of {len(models_to_run)} model(s)...")
    
    for repo_id, model_name in models_to_run:
        download_model(repo_id, model_name)
        
    print("\n🎉 Batch download process finished!")