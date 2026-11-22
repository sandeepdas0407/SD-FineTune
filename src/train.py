"""LoRA SFT trainer (config-driven). Works on the T4 VM (fp32 base + fp16 autocast) and on CPU (smoke test).

Usage:
  python src/train.py --config configs/uc2.yaml
  python src/train.py --config configs/uc2.yaml --max_steps 20 --limit 200          # smoke test
  python src/train.py --config configs/uc2.yaml --model trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --max_steps 2 --limit 16  # CPU check
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import to_prompt_completion  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def read_jsonl(p, limit=None):
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", help="override base model (e.g. tiny model for CPU checks)")
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--limit", type=int, help="use only the first N train/val rows")
    ap.add_argument("--output_dir", help="override output dir")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    t = cfg["train"]
    model_id = a.model or cfg["base_model"]
    out = Path(a.output_dir or ROOT / cfg["output_dir"])
    data = ROOT / cfg["data_dir"]
    cuda = torch.cuda.is_available()
    print(f"model={model_id} cuda={cuda} out={out}")

    train_ds = Dataset.from_list([to_prompt_completion(cfg["task"], r) for r in read_jsonl(data / "train.jsonl", a.limit)])
    val_ds = Dataset.from_list([to_prompt_completion(cfg["task"], r) for r in read_jsonl(data / "val.jsonl", a.limit)])

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # fp32 weights + fp16 autocast: T4 has no bf16; avoids fp16 overflow / "unscale FP16 gradients" errors
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, attn_implementation="sdpa")

    peft_cfg = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"],
                          target_modules="all-linear", task_type="CAUSAL_LM")

    args = SFTConfig(
        output_dir=str(out),
        num_train_epochs=t["epochs"],
        max_steps=a.max_steps,
        learning_rate=t["lr"],
        per_device_train_batch_size=t["per_device_batch"],
        per_device_eval_batch_size=t["per_device_batch"],
        gradient_accumulation_steps=t["grad_accum"],
        lr_scheduler_type="cosine",
        warmup_steps=t["warmup_ratio"],  # transformers 5: float in [0,1) = ratio of total steps
        logging_steps=10 if a.max_steps < 0 else 1,
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        save_strategy="steps",
        save_steps=t["save_steps"],
        save_total_limit=2,
        fp16=cuda,
        bf16=False,                  # TRL 1.x defaults bf16=True; T4 has no bf16
        use_cpu=not cuda,
        gradient_checkpointing=cuda,
        max_length=cfg["max_seq_len"],
        completion_only_loss=True,   # loss only on the assistant completion
        report_to="none",
        seed=t["seed"],
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                         processing_class=tok, peft_config=peft_cfg)
    trainer.model.print_trainable_parameters()

    last = get_last_checkpoint(str(out)) if out.exists() else None  # resume after interruption
    if last:
        print("resuming from", last)
    trainer.train(resume_from_checkpoint=last)

    adapter_dir = out / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    metrics = {"eval": trainer.evaluate(), "log_history": trainer.state.log_history}
    (out / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("saved adapter ->", adapter_dir)


if __name__ == "__main__":
    main()
