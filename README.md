# SD-FineTune: low-cost LoRA fine-tuning on an Azure T4 VM

This project fine-tunes **Qwen2.5-1.5B-Instruct** with LoRA on one Azure `Standard_NC4as_T4_v3` VM (eastus, about $0.53/hr), then compares outputs **before and after** fine-tuning on two customer-support use cases. See [PLAN.md](PLAN.md) for the full plan and its validation rounds.

**Repository:** https://github.com/sandeepdas0407/SD-FineTune

```bash
git clone https://github.com/sandeepdas0407/SD-FineTune.git
```

| Use case | Data | What improves |
|---|---|---|
| UC1: support replies | Bitext customer-support (27 intents, real) | House style, `{{Placeholder}}` use, structure (ROUGE-L, placeholder rate, Claude A/B judge) |
| UC2: banking triage JSON | Banking77 queries (real) + Claude-written replies (synthetic) | Intent accuracy, macro-F1, strict-JSON rate |

## Status
- [x] Data prepared: `data/processed/uc1` (3,996 / 216 / 270) and `uc2` (2,999 / 227 / 770), with no train/test leakage
- [x] Pipeline checked on CPU with a tiny model: training, eval in all 3 variants, report. Loss is computed only on the reply tokens.
- [ ] VM created → smoke test → baseline → train UC1/UC2 → report → teardown

## Run it
```powershell
# 0. Local (already done once)
python -m venv .venv; .venv\Scripts\pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install -r requirements-local.txt
python data/synth/make_banking77_replies.py; python data/prepare_uc1.py; python data/prepare_uc2.py

# 1. Create the VM (starts billing about $0.53/hr) + budget alert + SSH locked to your IP + nightly auto-shutdown
./azure/vm/create_vm.ps1 -AlertEmail <your-email>

# 1 (alternative). Same resources with Terraform; `terraform destroy` removes them all
cd azure/terraform; cp terraform.tfvars.example terraform.tfvars   # fill in subscription_id + alert_email
terraform init; terraform apply                                     # prints public_ip + ssh_command
```
```bash
# 2. Git Bash: push code, set up the GPU (the first run reboots, so run it twice)
bash azure/vm/sync.sh push
ssh azureuser@<ip> 'bash ~/SD-FineTune/azure/vm/setup_vm.sh'   # installs driver, reboots
ssh azureuser@<ip> 'bash ~/SD-FineTune/azure/vm/setup_vm.sh'   # venv + torch cu126 + libs

# 3. On the VM inside tmux:  tmux new -s ft
bash azure/vm/run.sh smoke       # about 5 min sanity check
bash azure/vm/run.sh baseline    # "pre": base zero-shot + 3-shot, UC1 + UC2
bash azure/vm/run.sh train uc2   # "post": LoRA + finetuned eval
bash azure/vm/run.sh train uc1

# 4. Locally: fetch results and build the report
bash azure/vm/sync.sh pull
.venv/Scripts/python src/make_report.py      # results/report.html, results/metrics.json, results/uc1/judge_pairs.jsonl
```

## Run locally with Ollama (CPU, free)
| Ollama model | Weights | System prompt |
|---|---|---|
| `sd-support-base` / `sd-support-ft` | Qwen2.5-1.5B base / + UC1 LoRA (merged) | UC1 support prompt |
| `sd-banking-base` / `sd-banking-ft` | Qwen2.5-1.5B base / + UC2 LoRA (merged) | UC2 triage prompt (77 labels, JSON) |

All four are GGUF Q8_0 (1.6 GB each) with temperature 0. They run at about 17–22 tok/s on an i5-12500H.
```bash
ollama run sd-banking-ft "the atm kept my card, what do I do now?"
python src/ollama_compare.py banking          # interactive side-by-side, base | fine-tuned
python src/ollama_compare.py support "how do I cancel order {{Order Number}}?"
```
`outputs/merged` and `outputs/gguf` were deleted after import on 2026-11-22 to free 13.4 GB. Ollama keeps its own copies in `%USERPROFILE%\.ollama\models\blobs`, and the LoRA adapters in `outputs/uc*/adapter` are kept.
To rebuild them: `python src/export_merged.py`, then llama.cpp `convert_hf_to_gguf.py outputs/merged/<m> --outtype q8_0 --outfile outputs/gguf/sd-<m>-q8_0.gguf` (b12122, needs `sentencepiece`), then `python src/make_ollama.py`.
Ollama 0.40.x imports safetensors only through its MLX engine, which doesn't support Qwen2. That's why the models go through GGUF.

## Cost control
- `az vm deallocate -g rg-sd-finetune -n vm-sd-ft` after every session. When deallocated, only the disk and IP are billed.
- Auto-shutdown runs nightly at 18:30 UTC, and budget alerts fire at $25 and $50.
- When finished: `az group delete -n rg-sd-finetune --yes` removes everything (or `terraform destroy` if you created it with Terraform).

## Layout
`src/prompts.py` holds the prompts shared by training and eval · `src/train.py` is the LoRA SFT trainer (fp32 base + fp16 autocast, resumes from checkpoints) · `src/evaluate.py` runs base / base_fewshot / finetuned · `src/metrics.py` · `src/make_report.py` · `azure/vm/*` holds the VM create, setup, run and sync scripts · `azure/terraform/*` is the Terraform version of `create_vm.ps1` · `configs/uc*.yaml` holds the hyperparameters.

## Version notes (found by the CPU smoke test)
- transformers 5.x removed `warmup_ratio`. Use `warmup_steps=<float in [0,1)>` instead.
- TRL 1.x defaults to `bf16=True`. It is forced to `False`, because a T4 has no bf16.

## Azure/VM notes (found during the first real run, 2026-11-21)
- **`--security-type Standard` was rejected.** It needs the subscription feature `Microsoft.Compute/UseStandardSecurityType`. The VM now uses **Trusted Launch with Secure Boot off**, so the NVIDIA module loads without signing.
- **`ubuntu-drivers install --gpgpu` installs a headless driver** on Azure kernels (595-server-open). It does not include `nvidia-smi` or `libcuda` and does not load the module at boot. `setup_vm.sh` now adds `nvidia-utils` and `libnvidia-compute`, plus `/etc/modules-load.d/nvidia.conf`, so no reboot is needed.
- **TRL 1.x's fused loss kernel needs Triton**, which compiles with gcc. `setup_vm.sh` installs `build-essential`.
- **Smoke-test throughput on the T4:** about 5–6 s per step (effective batch 16, UC2), so about 35–40 min per full training run.
