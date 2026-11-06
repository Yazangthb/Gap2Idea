#!/usr/bin/env bash
# Task 5a: our own Qwen server on GPU 1 of a100-learner, matching the funnel's server on GPU 0, port 8001:
# same model, revision, vLLM version and flags (minus tool calling, which these calls don't use).
#
#   setup    venv with vllm 0.32.0, download the pinned revision, make an API key (run once)
#   start    serve on GPU 1, port 8002, in tmux session qwen5a; refuses if GPU 1 is already in use
#   status   tmux session, /version and GPU memory
#   stop     end the session and free GPU 1
#
# From your PC:  ssh hamza@10.100.30.86 'bash -s' -- setup < scripts/bench/serve_qwen_5a.sh
# Then put QWEN_BASE_URL=http://10.100.30.86:8002/v1 and the key from ~/qwen5a/api_key in .env (never commit it).
set -euo pipefail

DIR="$HOME/qwen5a"
VENV="$DIR/venv"
KEY="$DIR/api_key"
LOG="$DIR/serve.log"
MODEL="Qwen/Qwen3.8-27B-FP8"
REV="017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
VLLM_VERSION="0.32.0"
GPU=1
PORT=8002
SESSION=qwen5a
export HF_HOME="$DIR/hf"

case "${1:-}" in
setup)
  mkdir -p "$DIR" && chmod 700 "$DIR"
  [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q "vllm==$VLLM_VERSION"
  "$VENV/bin/python" -c "from huggingface_hub import snapshot_download as d; print(d('$MODEL', revision='$REV'))"
  [ -s "$KEY" ] || (umask 077 && "$VENV/bin/python" -c "import secrets; print(secrets.token_urlsafe(32))" > "$KEY")
  "$VENV/bin/vllm" --version
  echo "setup done; API key is in $KEY"
  ;;
start)
  if [ ! -x "$VENV/bin/vllm" ] || [ ! -s "$KEY" ]; then echo "run setup first"; exit 1; fi
  used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "$used" -ge 1024 ]; then echo "GPU $GPU is in use ($used MiB); not starting"; exit 1; fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then echo "session $SESSION is already running"; exit 1; fi
  cat > "$DIR/serve.sh" <<EOF
export CUDA_VISIBLE_DEVICES=$GPU HF_HOME=$HF_HOME HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0
export VLLM_API_KEY="\$(cat $KEY)"
exec $VENV/bin/vllm serve $MODEL --revision $REV --served-model-name qwen3.8-27b \\
  --host 0.0.0.0 --port $PORT --max-model-len 262144 --max-num-seqs 64 --reasoning-parser qwen3 \\
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' 2>&1 | tee $LOG
EOF
  tmux new-session -d -s "$SESSION" "bash $DIR/serve.sh"
  echo "starting on GPU $GPU, port $PORT; loading takes a few minutes. Log: $LOG"
  ;;
status)
  tmux has-session -t "$SESSION" 2>/dev/null && echo "session $SESSION: running" || echo "session $SESSION: not running"
  curl -s -m 5 "http://localhost:$PORT/version" && echo || echo "port $PORT: not answering"
  nvidia-smi -i "$GPU" --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
  ;;
stop)
  tmux kill-session -t "$SESSION" 2>/dev/null && echo "stopped" || echo "not running"
  ;;
*)
  echo "usage: $0 setup|start|status|stop"; exit 2
  ;;
esac
