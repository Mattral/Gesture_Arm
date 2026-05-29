#!/usr/bin/env bash
# scripts/simulate_node_failure.sh
#
# Chaos test: pick 4 worker pods at random in the active TorchElastic run
# and SIGKILL them. Confirms the elastic harness:
#   1. Detects the loss via heartbeat / monitored_barrier timeout.
#   2. Tears down the dead ranks.
#   3. Re-rendezvouses surviving workers.
#   4. Reshards experts via ClusterStateMachine.reshard.
#   5. Reloads the latest async checkpoint from S3.
#   6. Resumes training without operator intervention.
#
# Usage (Kubernetes):
#   ./simulate_node_failure.sh                 # 4 random pods
#   ./simulate_node_failure.sh -n 8            # 8 random pods
#   ./simulate_node_failure.sh -l app=moe      # custom selector
#
# Usage (bare-metal / Slurm): pass a list of ranks to kill.
#   ./simulate_node_failure.sh -r 4,5,6,7

set -euo pipefail

NUM_TO_KILL=4
SELECTOR="app=moe-engine"
RANKS=""

while getopts "n:l:r:" opt; do
  case $opt in
    n) NUM_TO_KILL="$OPTARG" ;;
    l) SELECTOR="$OPTARG" ;;
    r) RANKS="$OPTARG" ;;
    *) echo "usage: $0 [-n num_pods] [-l label_selector] [-r 1,2,3]" >&2; exit 1 ;;
  esac
done

if [[ -n "$RANKS" ]]; then
  echo "[chaos] killing ranks: $RANKS"
  IFS=',' read -ra ARR <<< "$RANKS"
  for r in "${ARR[@]}"; do
    pid=$(pgrep -f "RANK=$r" || true)
    [[ -n "$pid" ]] && kill -9 "$pid"
  done
  exit 0
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "[chaos] kubectl not found; pass -r instead" >&2; exit 1
fi

PODS=$(kubectl get pods -l "$SELECTOR" -o name | shuf | head -n "$NUM_TO_KILL")
echo "[chaos] killing $NUM_TO_KILL pods:" $PODS
for pod in $PODS; do
  kubectl delete "$pod" --grace-period=0 --force --wait=false
done
echo "[chaos] kill sent. Watch for elastic recovery in the trainer logs."
