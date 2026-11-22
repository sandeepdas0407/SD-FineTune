#!/usr/bin/env bash
# Run LOCALLY (Git Bash).  push: send code + processed data to the VM.  pull: fetch results + adapters back.
#   bash azure/vm/sync.sh push
#   bash azure/vm/sync.sh pull
set -euo pipefail
cd "$(dirname "$0")/../.."
RG=${RG:-rg-sd-finetune}; VM=${VM:-vm-sd-ft}
IP=$(az vm show -d -g "$RG" -n "$VM" --query publicIps -o tsv | tr -d '\r')
[ -n "$IP" ] || { echo "VM has no public IP (is it running? az vm start -g $RG -n $VM)"; exit 1; }
SSH="ssh -o StrictHostKeyChecking=accept-new azureuser@$IP"
case "${1:-}" in
  push)
    tar czf - --exclude=.venv --exclude=.git --exclude=outputs --exclude=data/raw --exclude=results --exclude=logs . \
      | $SSH "mkdir -p ~/SD-FineTune && tar xzf - -C ~/SD-FineTune"
    echo "pushed to $IP" ;;
  pull)
    $SSH "cd ~/SD-FineTune && tar czf - results logs outputs/*/adapter outputs/*/train_metrics.json 2>/dev/null || true" | tar xzf -
    echo "pulled results/, logs/, outputs/*/adapter from $IP" ;;
  *) echo "usage: sync.sh push|pull"; exit 1 ;;
esac
