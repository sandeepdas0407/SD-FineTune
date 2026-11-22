#!/usr/bin/env bash
# Run ON THE VM (inside tmux so an SSH drop doesn't kill the job).
#   bash azure/vm/run.sh smoke        # ~5 min: 20 train steps + 20 eval samples on UC2
#   bash azure/vm/run.sh baseline     # base + base_fewshot for UC1 and UC2  (the "pre")
#   bash azure/vm/run.sh train uc1    # LoRA train + finetuned eval          (the "post")
#   bash azure/vm/run.sh train uc2
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
mkdir -p logs
case "${1:-}" in
  smoke)
    $PY src/train.py --config configs/uc2.yaml --max_steps 20 --limit 200 --output_dir outputs/smoke_uc2 2>&1 | tee logs/smoke_train.log
    $PY src/evaluate.py --config configs/uc2.yaml --variants base,finetuned --adapter outputs/smoke_uc2/adapter \
        --limit 20 --results_dir outputs/smoke_results/uc2 2>&1 | tee logs/smoke_eval.log ;;
  baseline)
    for uc in uc1 uc2; do
      $PY src/evaluate.py --config configs/$uc.yaml --variants base,base_fewshot 2>&1 | tee logs/baseline_$uc.log
    done ;;
  train)
    uc=${2:?usage: run.sh train uc1|uc2}
    $PY src/train.py --config configs/$uc.yaml 2>&1 | tee logs/train_$uc.log
    $PY src/evaluate.py --config configs/$uc.yaml --variants finetuned --adapter outputs/$uc/adapter 2>&1 | tee logs/eval_ft_$uc.log ;;
  *) echo "usage: run.sh smoke|baseline|train <uc>"; exit 1 ;;
esac
