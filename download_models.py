import os
import sys
from huggingface_hub import snapshot_download

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "hf_cache")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR

print(f"Setting HF Cache dir to: {CACHE_DIR}")

# 1. Download adapter Nihilux/Valkeryne/final_model
adapter_dir = os.path.join(BASE_DIR, "valkeryne_lora")
print("Downloading Nihilux/Valkeryne final_model...")
snapshot_download(
    repo_id="Nihilux/Valkeryne",
    allow_patterns=["final_model/*"],
    local_dir=adapter_dir,
    cache_dir=CACHE_DIR
)
print("Valkeryne adapter downloaded successfully!")

# 2. Download base model unsloth/Qwen3.5-0.8B
base_model_dir = os.path.join(BASE_DIR, "qwen3.5_base")
print("Downloading unsloth/Qwen3.5-0.8B...")
snapshot_download(
    repo_id="unsloth/Qwen3.5-0.8B",
    local_dir=base_model_dir,
    cache_dir=CACHE_DIR
)
print("Base model downloaded successfully!")
