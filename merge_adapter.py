import os
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

MODEL_ID   = "Qwen/Qwen3-4B-Thinking-2507"
ADAPTER    = "/home/asdu/151B_SP26_Competition/sft_checkpoints/final_adapter"
MERGED_DIR = "/home/asdu/151B_SP26_Competition/sft_merged"

Path(MERGED_DIR).mkdir(parents=True, exist_ok=True)

# Move .local to scratch to free up home directory space
print("Freeing home directory space...")
shutil.move("/home/asdu/.local", "/scratch/local_backup")

try:
    print("Loading base model on CPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print("Loading adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER)

    print("Merging...")
    model = model.merge_and_unload()

    print("Saving...")
    model.save_pretrained(MERGED_DIR, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
    tokenizer.save_pretrained(MERGED_DIR)

    print(f"Done. Merged model saved to {MERGED_DIR}")

finally:
    # Always restore .local even if something fails
    print("Restoring .local...")
    shutil.move("/scratch/local_backup", "/home/asdu/.local")
    print("Restored.")
