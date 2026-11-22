# Plan: Low-cost LoRA fine-tuning of a small open-weight model on Azure ML (driven from Claude Code + VS Code)

## Context
You want to fine-tune an open-weight LLM on Azure for as little money as possible, using public online data plus some synthetic data, all driven from Claude Code inside VS Code. The goal is a clear before/after comparison of model output. What you chose: pay-as-you-go subscription with a **< $100** budget, **self-managed LoRA on an Azure ML compute cluster**, **domain Q&A / customer-support** use cases, and a **small (1–4B) model**.

Environment check (read-only): `SD-FineTune/` is empty and not a git repo. Python 3.11.3, Azure CLI and VS Code are installed. You're logged in to "Azure subscription 1". The Azure CLI `ml` extension is **not** installed yet (only `quota` and `application-insights` are).

## Key decisions
| Item | Choice | Why |
|---|---|---|
| Base model | **Qwen2.5-1.5B-Instruct** (Apache-2.0, not gated) | Cheap and fast on a T4. Works in fp16/fp32 on a T4. Small models show the largest before/after gain. |
| Stretch model | Phi-4-mini-instruct (3.8B, MIT) with QLoRA | Optional re-run of the best use case to compare model sizes |
| Region | **`eastus`** | It's the only region checked where your subscription already has T4 VM quota (8 vCPUs). It also has the lowest T4 prices: $0.526/hr dedicated and $0.105/hr low-priority. |
| Compute | **DECIDED (Round 4): a single `Standard_NC4as_T4_v3` VM (1× T4 16 GB) in eastus, Ubuntu 22.04**, using your existing 8-vCPU T4 VM quota. Dedicated at $0.526/hr, auto-shutdown every night, deallocated after each session. Driven from VS Code Remote-SSH and Claude Code. The Azure ML cluster is now only an alternative, if its quota is ever granted. | It's available today with no quota wait. GPU cost is about $3, about $5 in total. |
| Method | LoRA (r=16, alpha=32, dropout 0.05, all linear layers), base loaded in fp32, fp16 autocast, gradient checkpointing, SDPA attention | A T4 has no bf16 and no FlashAttention-2. Loading the base in fp32 avoids fp16 overflow and the "unscale FP16 gradients" errors. |
| Training lib | HF `transformers` + `peft` + `trl` (SFTTrainer) with **prompt/completion conversational format** | Loss is computed only on the assistant completion. This needs no `{% generation %}` tags in the chat template, which Qwen2.5 doesn't have. |
| Job submission | Azure ML **CLI v2 YAML** (`az ml job create --stream`) | Easy for Claude Code to run, watch and download from the terminal |
| Pre/post judge | Automatic metrics plus a **blind A/B judgement by Claude Code** on a sample | No extra API key or cost |
| Serving | No online endpoint. Inference runs as a batch eval job on the same cluster. | Avoids always-on endpoint costs |
| Job environment | **Use a public prebuilt image directly** (`pytorch/pytorch` CUDA 12.x runtime, exact tag pinned) and `pip install -r requirements.txt` when the job starts. No custom image build. | No image build means no Azure Container Registry is created, which saves about $5/month. The pip install adds about 1–2 GPU-minutes per job, roughly $0.01–0.02. |
| Job granularity | **One combined job per stage**: one baseline-eval job for both UCs, one train+eval job per UC | Fewer cold starts, so the image pull, model download and pip install happen less often |

## Use cases (2 core + 1 optional)

### UC1: Customer-support response assistant (public data)
- **Data:** `bitext/Bitext-customer-support-llm-chatbot-training-dataset` on Hugging Face. About 26.9k rows across 27 intents and 11 categories. Fields: `instruction`, `intent`, `category`, `response`. License: CDLA-Sharing-1.0.
- **Task:** customer message → reply in a consistent house style: empathetic opening, numbered steps, `{{Placeholder}}` slots such as `{{Order Number}}`, and a closing line.
- **Splits:** stratified by intent. Train 4,000 · val 200 · test 270 (10 per intent). Exact and near-duplicate `instruction`s are removed across splits, because Bitext contains paraphrase variants.
- **Metrics:** ROUGE-L against the reference, placeholder-adherence rate, rate of including numbered steps, length ratio, and a Claude blind A/B win-rate on 30 test samples.

### UC2: Banking support triage: intent + reply as JSON (public + synthetic data)
- **Data:** Banking77, 77 intents, CC-BY-4.0, with 10,003 train and 3,080 test rows. **Fetch it from the PolyAI GitHub CSVs** (`PolyAI-LDN/task-specific-datasets/banking_data/{train,test}.csv`), because the HF `PolyAI/banking77` loader is script-based and recent `datasets` versions refuse to run it.
- **Synthetic part:** Claude Code writes `data/synth/banking77_replies.jsonl`: 3 reply variants for each of the 77 intents (231 replies, generic and safe, no fake account data). Each real customer query is paired with a random variant for its intent. The queries are real; the responses are synthetic.
- **Task:** output strict JSON `{"intent": "<label>", "reply": "<text>"}`. The system prompt lists all 77 labels for **both** the base and the fine-tuned model, so the comparison is fair.
- **Splits:** train 3,000 (about 39 per intent, stratified) · val 231 · test 770 (10 per intent from the official test split).
- **Metrics:** intent accuracy, macro-F1, JSON-valid rate, label-in-set rate. These are objective and show the gain most clearly.

### UC3 (optional stretch): Azure ML docs helpdesk (synthetic Q&A)
- Claude Code reads a few pages from the public `MicrosoftDocs/azure-ai-docs` repo (CC-BY-4.0) and generates about 400 Q&A pairs. 50 are held out for testing.
- This shows the **limits** of fine-tuning: it learns style well but learns facts poorly. Expect smaller gains. Compare against "base + doc pasted in prompt" to show why RAG is usually better for facts.

## Baselines for the "pre" side (each scored on the same test set)
1. **Base zero-shot**: the same system prompt as the fine-tuned model.
2. **Base 3-shot**: the same prompt plus 3 training examples. This is a fair, strong baseline.
3. **Fine-tuned (LoRA)**: zero-shot.

All three use greedy decoding (`do_sample=False`), the same `max_new_tokens` (UC1: 300, UC2: 150) and the Qwen chat template.

## Project layout (to be created in `SD-FineTune/`)
```
SD-FineTune/
  README.md                       # how to run each step
  .gitignore                      # data/raw, outputs, .venv
  requirements-local.txt          # datasets, pandas, scikit-learn, rouge-score, pyyaml
  azure/
    setup.ps1                     # RG + workspace + cluster + budget (idempotent)
    compute.yml                   # amlcompute T4, min 0/max 1, low_priority, idle 300s
    env/requirements.txt          # pinned: transformers, peft, trl, accelerate, datasets, (bitsandbytes for stretch)
    env/environment.yml           # image-only env (public pytorch image, no build → no ACR)
    jobs/baseline.yml             # command job: pip install → evaluate.py base + base_fewshot for UC1+UC2
    jobs/train_eval.yml           # command job: pip install → train.py → evaluate.py finetuned (per UC)
  configs/uc1.yaml, uc2.yaml, (uc3.yaml)   # model id, LoRA, lr, epochs, seq len, prompts
  data/
    prepare_uc1.py                # HF download → dedupe → stratified split → prompt/completion JSONL
    prepare_uc2.py                # GitHub CSV → pair with synth replies → JSONL
    synth/banking77_replies.jsonl # Claude-authored
    processed/uc1/, uc2/          # train/val/test.jsonl (uploaded automatically as job inputs)
  src/
    prompts.py                    # system prompts + few-shot builder (shared by train & eval)
    train.py                      # generic SFT/LoRA trainer (config-driven, checkpoint + resume)
    evaluate.py                   # batched generation for 3 variants → predictions.jsonl
    metrics.py                    # ROUGE-L, placeholder/step adherence, JSON validity, acc/F1
    make_report.py                # metrics table + side-by-side examples → results/report.html
  results/                        # downloaded predictions, metrics.json, report.html, judge.md
```
`src/prompts.py` is the single source of truth for prompts. Training and eval both import it, so the prompt can never differ between them.

## Execution steps

**Phase 0: Azure prerequisites (do first, because quota approval can take hours to 1–2 days)**
1. `az extension add -n ml`
2. Region: **`eastus`** (verified; see Round 3).
3. In the Azure portal, go to **Quotas → Azure Machine Learning → eastus** and request:
   - **Standard NCASv3_T4 Family Cluster Dedicated vCPUs: 0 → 4**
   - **Total Cluster Low Priority Regional vCPUs: 0 → 4.** The NCASv3_T4 low-priority family limit is already −1 (unlimited), so only this regional total is blocking.

   Azure ML quota is **separate** from VM quota, and both of these are currently 0. Recheck with `az ml compute list-usage -l eastus` (after step 1).
   - If both are approved, use `low_priority` ($0.105/hr). If only dedicated is approved, use `dedicated` ($0.526/hr), which is still about $3 in total.
   - If neither is approved within about 2 days, switch to the **VM fallback** (Phase 0b).

4. In `azure/setup.ps1`: create resource group `rg-sd-finetune`, create workspace `mlw-sd-finetune`, create the compute cluster from `compute.yml`, and set a **Cost Management budget alert at $25 and $50** on the resource group.

**Phase 0b: VM fallback (only if the Azure ML quota is not approved)**
- Create an Ubuntu 22.04 `Standard_NC4as_T4_v3` VM in eastus. It uses your **existing 8-vCPU VM quota**, so it's available today. Use dedicated pricing: Spot needs 4 low-priority VM vCPUs and you have 3.
- Install the NVIDIA driver with the GPU driver VM extension, give it a 64 GB Standard SSD OS disk, and turn on **auto-shutdown** (free) every night.
- Connect from VS Code Remote-SSH. Claude Code runs the same `train.py` / `evaluate.py` scripts over SSH, and the `azure/jobs/*.yml` files are not needed.
- **Deallocate** the VM after each session (`az vm deallocate`). When deallocated, only the disk is billed, about $5/month prorated.
- Cost: about 6 hr × $0.526 ≈ $3 GPU, plus about $1–2 for the disk and public IP. That's about $5 in total.

**Phase 1: Local setup in VS Code**
5. `git init`, create a `.venv` with Python 3.11, and `pip install -r requirements-local.txt`. The local machine doesn't need torch or a GPU.
6. Optional: install the "Azure Machine Learning" VS Code extension to browse jobs.

**Phase 2: Data**
7. Run `python data/prepare_uc1.py` and check the counts per split and per intent.
8. Claude Code writes `data/synth/banking77_replies.jsonl`. You spot-check about 10 replies. Then run `python data/prepare_uc2.py`.
9. Sanity check: no test instruction appears in train, every label in the test split exists in train, and the token-length distribution fits `max_seq_len` (UC1: 768, UC2: 1024).

**Phase 3: Environment + smoke test**
10. `az ml environment create -f azure/env/environment.yml`. It only references an image and builds nothing, so no ACR is created. Pin compatible library versions in `requirements.txt` at implementation time.
11. Run a **smoke job**: `train_eval.yml` on UC2 with `max_steps=20` and 20 eval samples. This catches environment, CUDA and prompt bugs for a few cents before the full runs.

**Phase 4: Baseline evaluation (the "pre")**
12. Run **one** `baseline.yml` job that covers UC1 and UC2 with the variants `base` and `base_fewshot`. Download the results with `az ml job download` into `results/<uc>/`.

**Phase 5+6a: Fine-tune and evaluate the FT model in the same job**
13. Run `train_eval.yml` for UC1: 2 epochs, lr 2e-4, cosine schedule, warmup 3%, effective batch 16, eval on val every 100 steps, checkpoint every 100 steps with automatic resume (in case a low-priority node is preempted). The same job then runs `evaluate.py --variant finetuned` on the test set.
14. Run `train_eval.yml` for UC2 with the same hyperparameters.
15. Outputs: the LoRA adapter (about 20–40 MB) and `predictions_finetuned.jsonl`.

**Phase 6: Comparison**
16. Download the outputs of both train_eval jobs into `results/<uc>/`.
17. Run `python src/make_report.py` to produce `results/report.html` and `results/metrics.json`. The report contains a metrics table for each UC (zero-shot / 3-shot / FT), deltas, and 10 side-by-side examples per UC.
18. Claude Code does a blind A/B judgement on 30 UC1 samples with randomized order and a rubric (helpfulness, tone, structure, placeholder use) and writes `results/judge.md`.
19. Optional: publish the report as a private Artifact so you can share it.

**Phase 7: Stretch + teardown**
20. Optional stretch: Phi-4-mini with QLoRA (nf4, fp16 compute) on UC2, and/or UC3.
21. **Teardown:** download the adapters and results, then run `az group delete -n rg-sd-finetune --yes`. This stops every remaining cost (storage, Key Vault, App Insights).

## Cost estimate (verify on the Azure pricing calculator for your region)
| Item | Est. usage | Est. cost |
|---|---|---|
| T4 GPU (NC4as_T4_v3 in eastus: **$0.105/hr low-priority**, $0.526/hr dedicated, live prices 2026-11-21) | about 4–6 GPU-hr total (smoke, 1 baseline job, 2 train+eval jobs, reruns) | **$0.5–0.7 low-priority / $2–3.2 dedicated** |
| Image build / ACR | none (image-only environment) | $0 |
| Storage, Key Vault, App Insights | small | < $1 |
| Synthetic data (Claude Code, existing subscription) | – | $0 |
| **Total** | | **≈ $2–6**, far under the $100 cap |

## Round 2 validation: is this the cheapest option on Azure?
| Alternative | Approx. cost for this project | Verdict |
|---|---|---|
| **Azure ML cluster, T4, low-priority, scale to 0 (chosen)** | about $1–4 GPU + less than $1 overhead | **Cheapest viable option.** It bills per second only while a job runs and can't be left running by accident. |
| Azure ML *serverless* job with spot tier | same GPU rate | Same cost; the cluster is kept because it makes the idle and tier settings explicit. Either is fine. |
| Azure GPU VM (NC4as_T4_v3, Spot) | Similar per hour, plus the OS disk (about $5–10/month) and an always-on risk if you forget to stop it | More expensive in practice and riskier. Rejected. |
| Azure AI Foundry serverless fine-tuning (Phi/Llama) | Per-token training cost plus **hourly hosting** of the FT deployment just to evaluate it | Several times more expensive. It also limits model and region choice. Rejected. |
| Azure OpenAI fine-tuning | Not open-weight, and hosting is billed hourly | Out of scope. Rejected. |
| Bigger GPU (A10/A100) | $3–4+/hr | Unnecessary for 1.5B LoRA. Rejected. |
| Smaller or partial GPUs (NVadsA10 fractional, CPU) | – | Fractional vGPU memory is too small or unsupported for training, and CPU training is impractically slow. **The T4 is the cheapest viable GPU.** |

**Changes made after round 2 validation**
1. **Removed the custom Docker build.** Jobs use a public image directly plus a runtime pip install, so no ACR exists. This saves about $5/month and the image build compute.
2. **Merged jobs.** Baseline eval for both UCs now runs in one job, and the FT eval runs inside the training job. That is 3 cold starts instead of 5, and about 1–2 fewer GPU-hours.
3. **Region:** prefer a US region where T4 quota is granted (`eastus` / `southcentralus` are usually cheapest for NCasT4_v3). Confirm in the pricing calculator.
4. **Low-priority first, dedicated as fallback.** Even fully dedicated, the GPU cost stays under about $4.
5. **Free safeguards kept:** a budget alert, min nodes = 0, a 300 s idle scale-down, and deleting the resource group at the end.

## Round 4: route decision (2026-11-21)
A full quota re-check found that the only **usable** GPU quota is **NCASv3_T4 = 8 vCPUs (VM) in eastus**. The other GPU quotas can't be used:
- NC / NC Promo (VM 12) and NC (Azure ML cluster 6) are K80 machines, retired in 2023.
- NV / NV Promo and NV (Azure ML cluster 6) are M60 machines, also retired.
- NVSv4 is an AMD fractional GPU without CUDA.
- Azure ML T4 cluster quota is 0.

**Decision (user): use the T4 VM now.** What changes in the plan:
- Phase 0 steps 3–4 (Azure ML quota, workspace, cluster) are **skipped**. **Phase 0b becomes the main route.**
- `azure/compute.yml`, `azure/env/environment.yml` and `azure/jobs/*.yml` are replaced by `azure/vm/create_vm.ps1`, `azure/vm/setup_vm.sh` and `azure/vm/run_*.sh`.
- The same `src/train.py` / `src/evaluate.py` scripts run on the VM inside a Python venv, with torch built for CUDA 12.6.
- Cost safeguards: budget alert on the resource group, nightly auto-shutdown (`az vm auto-shutdown`), `az vm deallocate` after each session, and `az group delete` at the end.

## Round 3 validation: checked against your live subscription, current SKUs and current prices (2026-11-21)
**What was checked (all read-only)**
| Check | Result | Impact on plan |
|---|---|---|
| `Standard_NC4as_T4_v3` offered (`az vm list-skus`) | Available with no restrictions in eastus, eastus2 and southcentralus (zones 1–3) | SKU is valid ✓ |
| VM quota (`az vm list-usage`) | **eastus:** NCASv3_T4 = **8**, low-priority VM = 3, regional total = 18. southcentralus and centralindia: T4 = 0. A10/A100 families = 0 everywhere. | Region changed to **eastus**. Spot VMs are not possible (3 < 4). The VM fallback must be dedicated. |
| **Azure ML quota** (Microsoft.MachineLearningServices usages) | eastus, eastus2, westus3: T4 cluster dedicated = **0**, low-priority regional total = **0**, dedicated regional total = 20 | **This blocks the original plan.** An explicit Azure ML quota request is needed (Phase 0, step 3), and the VM fallback was added. |
| Resource provider | `Microsoft.MachineLearningServices` is registered | No action needed ✓ |
| Retail prices (Azure Retail Prices API, Linux) | NC4as_T4_v3 in eastus: **$0.526 dedicated / $0.164 Spot VM / $0.105 low-priority**. Other regions: dedicated $0.526–0.658, low-priority $0.105–0.132. | Cost estimates confirmed. eastus is in the cheapest tier. |
| Models (HF API) | Qwen2.5-1.5B-Instruct: Apache-2.0, not gated. Phi-4-mini-instruct: MIT, not gated. | No HF token needed ✓ |
| Bitext dataset | Available on HF, CDLA-Sharing-1.0, single CSV `Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv` | `prepare_uc1.py` reads this CSV directly ✓ |
| Banking77 CSVs (PolyAI GitHub) | `train.csv` and `test.csv` both return HTTP 200 | Download source confirmed ✓ |
| Docker image | `pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime` (Jul 2026). The CUDA 12.6 builds still include the T4's architecture (sm_75). | **Pinned this tag** in `environment.yml` and the VM setup |

**Changes made after round 3**
1. The region is set to **eastus**, based on the measured quota and prices.
2. Added an explicit **Azure ML quota request**. Both T4 cluster quotas are 0 today, which would have made the first job fail.
3. Added **Phase 0b, a VM fallback** that uses the T4 VM quota you already have, so there's no waiting.
4. Removed Spot VM as an option: your low-priority VM quota is 3 vCPUs and NC4as needs 4.
5. Pinned the container image to `pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime`.
6. Confirmed the cost estimate with live prices: Azure ML low-priority about $1 total, Azure ML dedicated about $3, VM fallback about $5.

## First action after approval
Save this validated plan as **`SD-FineTune/PLAN.md`** (the markdown file you asked for) before starting Phase 0.

## Verification
- **Smoke test** (step 11) passes: loss decreases and eval writes valid predictions.
- **Data checks** (step 9): no train/test leakage and label sets match.
- **Training health:** val loss falls and doesn't climb by epoch 2. Check the job logs and metrics in Azure ML studio.
- **Expected outcome:** UC2 intent accuracy and JSON-valid rate rise clearly over base zero-shot and 3-shot. UC1 placeholder adherence, structure rate and ROUGE-L rise, and the Claude A/B win-rate for FT is above 50%. If FT doesn't beat 3-shot, the report says so honestly.
- **Cost check:** the cluster shows 0 nodes when idle (`az ml compute show`), and Cost Management shows spend within budget.

## Validation pass: issues found in the first draft and corrected
1. **Qwen2.5-3B license:** it uses the *Qwen Research License*, not Apache. Switched the base model to Qwen2.5-1.5B-Instruct (Apache-2.0). The stretch model is Phi-4-mini (MIT).
2. **T4 limits:** no bf16 and no FlashAttention-2. Changed to fp32 base + fp16 autocast with SDPA attention. Removed any bf16/FA2 settings.
3. **Completion-only loss:** TRL's `assistant_only_loss` needs generation tags in the chat template, and Qwen2.5 doesn't have them. Switched to the prompt/completion dataset format.
4. **Banking77 loader:** the HF dataset is script-based and breaks on newer `datasets`. Switched to the official PolyAI GitHub CSVs.
5. **Bitext leakage:** it has many paraphrase variants of the same message. Added dedup and near-duplicate filtering across splits, and the report notes this limitation.
6. **Unfair baseline:** comparing FT against zero-shot alone overstates gains. Added a 3-shot baseline, and the same label list and prompt are used for every variant.
7. **Low-priority preemption:** added checkpointing and resume, and a fallback to `dedicated` (still within budget).
8. **Hidden costs:** ACR is created on the first environment build. Added the ACR cost, a budget alert, and full teardown via deleting the resource group.
9. **Quota lead time:** new pay-as-you-go subscriptions often have 0 GPU quota. Moved the quota request to step 1 of execution.
10. **Prompt drift:** prompts are centralized in `src/prompts.py` and shared by training and eval.
11. **UC3 expectations:** fine-tuning is poor at injecting facts. UC3 is marked optional, and a "doc-in-prompt" comparison was added so it isn't oversold.
12. **Prices:** shown as estimates to check against the calculator, since they vary by region and over time.
