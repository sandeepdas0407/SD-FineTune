"""Export base + merged (base + LoRA) models as HF safetensors for Ollama import. CPU-only, bf16 to save RAM.

Usage: python src/export_merged.py            -> outputs/merged/{base,uc1,uc2}/
"""
import gc
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
BASE = "Qwen/Qwen2.5-1.5B-Instruct"
OUT = ROOT / "outputs" / "merged"


def save(model, tok, d: Path):
    d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(d, safe_serialization=True)
    tok.save_pretrained(d)
    print("saved", d, f"{sum(f.stat().st_size for f in d.glob('*.safetensors')) / 1e9:.2f} GB")


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    for name in ["base", "uc1", "uc2"]:
        model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, low_cpu_mem_usage=True)
        if name != "base":
            adapter = ROOT / "outputs" / name / "adapter"
            cfg = json.loads((adapter / "adapter_config.json").read_text())
            print(name, "adapter r=", cfg["r"], "targets=", cfg["target_modules"])
            model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
        save(model, tok, OUT / name)
        del model
        gc.collect()


if __name__ == "__main__":
    main()
