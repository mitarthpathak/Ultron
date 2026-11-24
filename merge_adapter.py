#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model and save a plain Hugging Face model,
ready for llama.cpp's convert_hf_to_gguf.py.

    python3 merge_adapter.py Qwen/Qwen3-4B-Instruct-2507 ultron-lora-topup/final ultron-merged

Needs about 9 GB of free memory (the model is loaded in bfloat16).
"""
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    base, adapter, out = sys.argv[1:4]
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)
    print("merged ->", out)


if __name__ == "__main__":
    main()
