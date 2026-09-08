#!/usr/bin/env bash
# agentsoscar.sh -- run a terminal coding agent (Claude Code or opencode) locally against
# an Ollama server on an Oscar GPU node. Supersedes opencode-oscar.sh: same core (Slurm
# allocation, safety gate, tunnel, readiness check) plus --backend/--model flags instead
# of hardcoded/commented-out choices.
#
# Usage:
#   ./agentsoscar.sh [--backend opencode|claude] [--model <ollama-model-ref>] [--ctx <tokens>] [--purge-cache]
#
# First-time setup (once per computer):
#   1. Install the agent you want. opencode is the default:
#        curl -fsSL https://opencode.ai/install | bash
#      Check it worked:  opencode --version
#   2. Create an SSH config entry named exactly "oscar-campus" on this computer:
#        https://docs.ccv.brown.edu/oscar/connecting-to-oscar/ssh/ssh-configuration-file
#      Check it worked:  ssh oscar-campus   (off campus, connect the Brown VPN first)
#   3. Put this script somewhere on your PATH and make it executable:
#        mkdir -p ~/.local/bin
#        cp agentsoscar.sh ~/.local/bin/
#        chmod +x ~/.local/bin/agentsoscar.sh
#      If `agentsoscar.sh --help` says "command not found", ~/.local/bin isn't on your
#      PATH yet; add this to ~/.zshrc (or ~/.bashrc) and open a new terminal:
#        export PATH="$HOME/.local/bin:$PATH"
#   You can also just run it in place with ./agentsoscar.sh after chmod +x.
#
# Examples:
#   ./agentsoscar.sh                                    # opencode + qwen3.8:27b (defaults, 256k ctx)
#   ./agentsoscar.sh --backend claude                   # Claude Code + qwen3.8:27b (breaks on long
#                                                        # conversations against Ollama -- see BACKEND=)
#   ./agentsoscar.sh --backend claude --model qwen3-coder:30b
#   ./agentsoscar.sh -b claude -m qwen3.6:35b --ctx 131072   # previous default, still available
#
# Self-healing: if there's no live connection to Oscar yet, this opens one itself
# (off campus, with the VPN connected, that may prompt Duo; on campus it's silent).
# If you don't even have an "oscar-campus" SSH config entry, it tells you exactly
# where to set one up instead of failing confusingly.
#
# Safe to run more than once, and safe to run alongside OTHER agents/sessions that
# might also be using Oscar right now: every session/job name this script creates is
# unique to this run (PID + timestamp), and on exit it only tears down resources it
# itself created -- if it found an existing Oscar connection already open (yours, or
# someone/something else's), it leaves that alone.
#
# The GPU allocation + ollama server run detached on the Oscar LOGIN node (setsid +
# nohup, no tmux/screen needed -- checked, tmux isn't on PATH there without a module
# load, so this avoids that dependency entirely) so they survive your ssh connection
# dropping. Progress/errors on the remote side land in a log file you can just `cat`,
# no terminal-multiplexer knowledge required. Tears everything down (tunnel + Slurm
# job + local connection holder if this run created one) when the agent exits -- Ctrl-C,
# normal exit, or this script killed.
#
# The model cache in /oscar/scratch is left in place across runs on purpose (so you
# don't re-download every time). Pass --purge-cache to also delete it on exit.
set -euo pipefail

REMOTE_HOST="oscar-campus"
# Oscar's ssh banner (PQ-warning + Brown consent notice) and a broken per-login
# `mkdir /user/<you>/scratch` (repeats ~14x on every single fresh connection --
# confirmed by testing one trivial `ssh oscar-campus echo`, not a frequency-of-calls
# thing) are pure noise. Not ours to fix -- read-only Oscar access, and it isn't in
# ~/.bashrc/.profile or the common /etc/profile.d/Lmod locations either. Filtered once,
# here, rather than at every call site; real stderr from the remote command or ssh
# itself still passes through untouched.
ssh_quiet() {
  ssh "$@" 2> >(grep -Ev \
    'post-quantum key exchange|store now, decrypt later|server may need to be upgraded|^#|mkdir: cannot create directory .*/user/.*/scratch' \
    >&2)
}
RUN_ID="$$_$(date +%s)"                 # unique to this invocation
CONN_HOLDER="opencode-connlink-$RUN_ID" # LOCAL tmux session (this Mac)
JOB_NAME="opencode-ollama-$RUN_ID"      # Slurm job name (so squeue/scancel can't hit a stale/unrelated job)
REMOTE_DIR=""                           # per-run staging dir; resolved after connecting (scp's remote
# path isn't shell-expanded, so \$(whoami) can't be used here --
# found by actually testing, not assumed)
PARTITION="gpu" # not gpu-debug: that QOS caps jobs at 1h (MaxWall=01:00:00,
# checked via `sacctmgr show qos`) -- too short for real sessions.
# `gpu`'s QOS (norm-gpu) has no wall-clock cap at all.
# GRES retry/cycle list -- this is a free/shared-pool script (no --account, deliberately:
# it has to work for users without condo access), and the shared gpu partition's L40S pool
# was directly confirmed saturated cluster-wide (checked `scontrol show node` AllocTRES on
# every L40S node: most at gres/gpu=8/8, a handful at 6-7/8 -- ~6 free GPUs total, scattered
# across nodes, out of ~90). That still beats the alternative 48GB-class type checked the
# same way (A6000: 2 free on one node, A40: 0 free anywhere) -- so L40S stays primary, just
# retried (a fresh salloc can land on a different, less-saturated node), with A6000/A40 as
# real fallbacks rather than hammering the same exhausted pool. RTX 3090/A5000/A5500/Quadro
# RTX 6000 are all 24GB, still not enough for qwen3.8:27b's ~35-40GB footprint at 256k ctx.
# We do NOT have confirmed proof of the exact kill mechanism (Slurm recorded these as a
# clean COMPLETED/exit-0, not the PREEMPTED state a forced kill usually shows) -- but retry
# is safe regardless of the exact mechanism, since the failure is empirically intermittent
# (one run survived 19+ minutes; others died in ~1-2s) and NOT tied to which model was
# requested (confirmed: qwen3.6:35b, the model that "was working fine," dies the identical
# way -- same node, same point in the script, before Ollama is even invoked).
GRES_OPTIONS=("gpu:l40s:1" "gpu:l40s:1" "gpu:nvidia_rtx_a6000:1" "gpu:l40s:1" "gpu:nvidia_a40:1")
GRES="${GRES_OPTIONS[0]}" # placeholder; actually assigned per-attempt in the retry loop below
CPUS=4
MEM="16G"
TIME_LIMIT="08:00:00" # self-imposed cap for a shared resource, not dictated by any QOS
# limit (gpu's QOS has none) -- keep sessions from running forever
# unattended. Agent-exit cleanup (scancel/tunnel teardown) still
# fires the same way regardless of this being longer now.

BACKEND="opencode" # "opencode" or "claude" -- toggled via --backend / -b. Default changed
# from "claude": confirmed via a captured request (local logging proxy, 2026-08-21) that
# Claude Code's Anthropic-Messages-API calls carry hundreds of role:"system" entries
# scattered through the message history (not just at index 0), which Ollama's Anthropic-
# compatible endpoint (/v1/messages) strictly rejects ("system message must be at the
# beginning") on any conversation with real history. opencode's OpenAI-compatible calls
# (/v1/chat/completions) hit a different, more permissive endpoint that has no such
# restriction -- verified working live (two real completions, ~1m20s each, zero errors)
# immediately after Claude failed identically 10/10 retries on the same server/model.
# This is specific to Ollama's endpoint, not a real Anthropic API limitation -- --backend
# claude remains available if Ollama's validator changes, or for short one-off sessions.
BASE_MODEL="qwen3.8:27b" # 27B DENSE (all params active every token, unlike qwen3.6:35b's
# MoE), 18GB weights download, Apache 2.0, released 2026-08-14 (confirmed: Qwen/Qwen3.8-27B
# on Hugging Face, and Ollama's own library page). Native 262,144-token context (matches
# NUM_CTX below exactly), extensible to 1M via YaRN. Adds vision input, unused by this
# script's text-only agent backends. Not yet run/measured on Oscar -- see the GRES comment
# above. Previous default qwen3.6:35b (35B MoE, ~3B active params, 73.4% SWE-bench
# Verified) still works via --model qwen3.6:35b if this one underperforms in practice.
# Toggle via --model / -m.
NUM_CTX=262144           # 256k tokens -- happens to equal qwen3.8:27b's native context
# exactly. Toggle via --ctx / --num-ctx.

# Oscar's `module load ollama` only offers 0.11.10/0.17.7/0.21.0 (0.21.0 default) --
# 0.21.0 crashed qwen3.6:35b at 128k context with a deterministic CUDA "illegal memory
# access" (confirmed real bug, not a VRAM issue: reported footprint was 28.2 GiB, well
# under the L40S's 48GB). Downloading a newer upstream release ourselves instead of using
# the module -- verified the compute nodes have internet access, the release asset
# resolves, and `tar` on Oscar supports .tar.zst natively (GNU tar 1.34). Pinned to
# v0.32.12 (released 2026-08-14) specifically because it's the release that ADDS qwen3.8:27b
# support (confirmed via the actual GitHub release + its asset list, not just a blog post
# claiming it) -- the previous pin, v0.32.9, predates this model and won't recognize it at
# all. Also required for the Claude Code backend regardless of model: Ollama's native
# Anthropic-Messages-API-compatible endpoint (confirmed via docs.ollama.com/integrations/claude-code)
# needs a build at least this recent.
OLLAMA_VERSION="v0.32.12"
# NOT the Ollama default (11434). gpu-debug/gpu nodes are multi-tenant (8 GPUs shared
# across users' jobs on the same node/network namespace) -- found by testing: a
# fresh allocation failed with "address already in use" on 11434, someone else's
# job on the same node had already bound the default port. Derive a per-run port
# instead of colliding with everyone's default.
OLLAMA_PORT=$((20000 + ($$ % 10000)))
PURGE_CACHE=false
OWN_CONN_HOLDER=false
TUNNEL_PID=""

show_help() {
  cat <<EOF
agentsoscar.sh -- Run a terminal coding agent locally against Ollama on an Oscar GPU node.

Usage:
  ./agentsoscar.sh [OPTIONS]

Options:
  -b, --backend <opencode|claude>   Agent to launch (default: opencode)
  -m, --model <model-tag>           Ollama base model to pull (default: qwen3.8:27b)
      --ctx, --num-ctx <tokens>     Context window token limit (default: 262144 = 256k)
      --purge-cache                 Delete remote model cache on exit
  -h, --help                        Show this help message and exit

GPU allocation: this is a free/shared-pool script (no Slurm account/condo). If a fresh
allocation dies within seconds of landing (a real, observed failure mode on the shared GPU
partition), it automatically retries on a different GPU type -- up to 5 attempts today
(l40s, l40s, a6000, l40s, a40), hard-capped at 10 regardless of how that list grows later.

Known limitation, --backend claude: Ollama's Anthropic-compatible endpoint rejects any
conversation whose message history has a system-role entry that isn't the very first
message -- a real constraint enforced by the model's own chat template, not a bug in this
script. Long conversations trip it reliably; --backend opencode (the default) goes through
Ollama's OpenAI-compatible endpoint instead, which has no such restriction. Use --backend
claude for short one-off asks, or if you specifically want Claude Code's own tooling.

Examples:
  ./agentsoscar.sh                                      # OpenCode + qwen3.8:27b (defaults, 256k ctx)
  ./agentsoscar.sh --backend claude                     # Claude Code -- fine for short asks, see above
  ./agentsoscar.sh --backend claude --model qwen3-coder:30b
  ./agentsoscar.sh -m qwen3.6:35b --ctx 131072           # previous default model, still available
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --purge-cache)
      PURGE_CACHE=true
      shift
      ;;
    --backend)
      if [ -z "${2:-}" ]; then
        echo "Error: --backend requires an argument ('opencode' or 'claude')" >&2
        exit 1
      fi
      BACKEND="$2"
      shift 2
      ;;
    --backend=*)
      BACKEND="${1#--backend=}"
      shift
      ;;
    -b)
      if [ -z "${2:-}" ]; then
        echo "Error: -b requires an argument ('opencode' or 'claude')" >&2
        exit 1
      fi
      BACKEND="$2"
      shift 2
      ;;
    -b=*)
      BACKEND="${1#-b=}"
      shift
      ;;
    --model)
      if [ -z "${2:-}" ]; then
        echo "Error: --model requires a model argument" >&2
        exit 1
      fi
      BASE_MODEL="$2"
      shift 2
      ;;
    --model=*)
      BASE_MODEL="${1#--model=}"
      shift
      ;;
    -m)
      if [ -z "${2:-}" ]; then
        echo "Error: -m requires a model argument" >&2
        exit 1
      fi
      BASE_MODEL="$2"
      shift 2
      ;;
    -m=*)
      BASE_MODEL="${1#-m=}"
      shift
      ;;
    --ctx|--num-ctx)
      if [ -z "${2:-}" ]; then
        echo "Error: $1 requires an integer token limit argument (e.g. 131072)" >&2
        exit 1
      fi
      NUM_CTX="$2"
      shift 2
      ;;
    --ctx=*|--num-ctx=*)
      NUM_CTX="${1#*=}"
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Run with --help for usage information." >&2
      exit 1
      ;;
  esac
done

if [ "$BACKEND" != "opencode" ] && [ "$BACKEND" != "claude" ]; then
  echo "Unknown --backend: '$BACKEND' (must be 'opencode' or 'claude')" >&2
  exit 1
fi

# Fail here rather than on the last line of the script. Without this, a missing agent is
# only discovered AFTER a GPU has been allocated, an 18GB model pulled and a tunnel opened
# -- the slowest possible way to find out you skipped step 1 of the setup above.
if ! command -v "$BACKEND" >/dev/null 2>&1; then
  cat >&2 <<EOF

'$BACKEND' is not installed, or not on this computer's PATH.

  opencode:     curl -fsSL https://opencode.ai/install | bash
  Claude Code:  https://docs.claude.com/en/docs/claude-code/setup

Check with "$BACKEND --version", then re-run this script. Nothing has been allocated on
Oscar yet, so there is nothing to clean up.
EOF
  exit 1
fi

# Ollama model names can't contain ':' or '/' as anything but the tag separator itself --
# derived deterministically from BASE_MODEL so --model doesn't need a matching --ctx-model.
CTX_K=$((NUM_CTX / 1024))
CTX_LABEL="${CTX_K}k"
CTX_MODEL="$(printf '%s' "$BASE_MODEL" | tr ':/' '--')-$CTX_LABEL"

cleanup() {
  echo ""
  echo "==> Cleaning up..."
  if [ -n "$TUNNEL_PID" ]; then
    kill "$TUNNEL_PID" 2>/dev/null || true
    wait "$TUNNEL_PID" 2>/dev/null || true
  fi
  if [ -n "$REMOTE_DIR" ]; then
    ssh_quiet "$REMOTE_HOST" "scancel --me --name=$JOB_NAME 2>/dev/null; sleep 2; rm -rf \"$REMOTE_DIR\"" || true
  else
    ssh_quiet "$REMOTE_HOST" "scancel --me --name=$JOB_NAME 2>/dev/null" || true
  fi
  if [ "$PURGE_CACHE" = true ]; then
    echo "==> Purging model cache from scratch (--purge-cache)..."
    ssh_quiet "$REMOTE_HOST" 'rm -rf /oscar/scratch/$(whoami)/ollama-models' || true
  fi
  if [ "$OWN_CONN_HOLDER" = true ]; then
    echo "==> Closing the connection to Oscar this run opened."
    tmux kill-session -t "$CONN_HOLDER" 2>/dev/null || true
  else
    echo "==> Leaving your existing Oscar connection open (this run didn't create it)."
  fi
  echo "==> Done. GPU allocation released."
}
trap cleanup EXIT INT TERM HUP

echo "==> Checking for a live connection to Oscar..."
if ssh_quiet -O check "$REMOTE_HOST" 2>/dev/null; then
  echo "==> Reusing an existing connection (yours, or another session's)."
else
  if ! ssh_quiet -G "$REMOTE_HOST" 2>/dev/null | grep -qi "^hostname "; then
    cat >&2 <<EOF

You don't have an SSH config entry named "$REMOTE_HOST" set up on this computer yet.
This script can't connect to Oscar until you do that once. Instructions here:

  https://docs.ccv.brown.edu/oscar/connecting-to-oscar/ssh/ssh-configuration-file

Once that's done (and you can run "ssh $REMOTE_HOST" successfully by hand), re-run this script.
EOF
    exit 1
  fi

  echo "==> Opening a new connection to Oscar (on campus this should be silent; off campus, with the VPN connected, you may get a Duo prompt now)..."
  tmux new-session -d -s "$CONN_HOLDER" "ssh $REMOTE_HOST"
  OWN_CONN_HOLDER=true

  CONNECTED=false
  for i in $(seq 1 30); do
    if ssh_quiet -o BatchMode=yes -o ConnectTimeout=5 "$REMOTE_HOST" true 2>/dev/null; then
      CONNECTED=true
      break
    fi
    sleep 2
  done

  if [ "$CONNECTED" != true ]; then
    cat >&2 <<EOF

Couldn't connect to Oscar after 60 seconds. Most likely causes:
  - If you're off Brown's campus network: connect the Brown VPN, then re-run this script.
  - If a Duo push appeared on your phone and you missed it: re-run this script and
    approve it promptly this time.
  - If neither applies, try running "ssh $REMOTE_HOST" by hand to see the actual error.
EOF
    exit 1
  fi
  echo "==> Connected."
fi

REMOTE_USER=$(ssh_quiet "$REMOTE_HOST" whoami 2>/dev/null)
if [ -z "$REMOTE_USER" ]; then
  echo "Could not determine your Oscar username (ssh $REMOTE_HOST whoami returned nothing)." >&2
  exit 1
fi
BASE_JOB_NAME="$JOB_NAME"
BASE_REMOTE_DIR="/oscar/scratch/$REMOTE_USER/opencode-runs/$RUN_ID"

cleanup_failed_attempt() {
  if [ -n "${TUNNEL_PID:-}" ]; then
    kill "$TUNNEL_PID" 2>/dev/null || true
    wait "$TUNNEL_PID" 2>/dev/null || true
    TUNNEL_PID=""
  fi
  ssh_quiet "$REMOTE_HOST" "scancel --me --name=$JOB_NAME 2>/dev/null" || true
}

MAX_ATTEMPTS=10 # hard cap regardless of how long GRES_OPTIONS grows; currently a no-op
# since GRES_OPTIONS has only 5 entries -- kept as a safeguard for future edits, not
# because 5 needed lowering.
NUM_ATTEMPTS=${#GRES_OPTIONS[@]}
[ "$NUM_ATTEMPTS" -gt "$MAX_ATTEMPTS" ] && NUM_ATTEMPTS=$MAX_ATTEMPTS

SUCCESS=false
for ((ATTEMPT_NUM = 0; ATTEMPT_NUM < NUM_ATTEMPTS; ATTEMPT_NUM++)); do
  GRES="${GRES_OPTIONS[$ATTEMPT_NUM]}"
  JOB_NAME="${BASE_JOB_NAME}-a${ATTEMPT_NUM}"
  REMOTE_DIR="${BASE_REMOTE_DIR}-a${ATTEMPT_NUM}"
  echo "==> Attempt $((ATTEMPT_NUM + 1))/${NUM_ATTEMPTS}: requesting $GRES"

echo "==> Preparing the remote launch script..."
LOCAL_LAUNCH_SCRIPT="$(mktemp)"
cat >"$LOCAL_LAUNCH_SCRIPT" <<LAUNCH
#!/usr/bin/env bash
export OLLAMA_MODELS=/oscar/scratch/\$(whoami)/ollama-models
mkdir -p "\$OLLAMA_MODELS"
exec salloc -p $PARTITION --gres=$GRES -c $CPUS --mem=$MEM --time=$TIME_LIMIT --job-name=$JOB_NAME srun bash -c '
  exec 9>"$REMOTE_DIR/trace.log"
  export BASH_XTRACEFD=9
  export PS4="+ \$(date +%T) "
  set -x
  echo "RUNNING ON: \$(hostname)" > "$REMOTE_DIR/ran_on.txt"
  OLLAMA_BIN_ROOT=/oscar/scratch/\$(whoami)/ollama-bin/$OLLAMA_VERSION
  if [ ! -x "\$OLLAMA_BIN_ROOT/bin/ollama" ]; then
    mkdir -p "\$OLLAMA_BIN_ROOT"
    TARBALL=/tmp/ollama-$OLLAMA_VERSION-\$\$.tar.zst
    curl -fsSL "https://github.com/ollama/ollama/releases/download/$OLLAMA_VERSION/ollama-linux-amd64.tar.zst" -o "\$TARBALL"
    tar --zstd -xf "\$TARBALL" -C "\$OLLAMA_BIN_ROOT"
    rm -f "\$TARBALL"
  fi
  export PATH="\$OLLAMA_BIN_ROOT/bin:\$PATH"
  export LD_LIBRARY_PATH="\$OLLAMA_BIN_ROOT/lib/ollama:\${LD_LIBRARY_PATH:-}"
  export OLLAMA_MODELS=/oscar/scratch/\$(whoami)/ollama-models
  MY_IP=\$(hostname -I | tr " " "\\n" | grep -E "^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+\$" | head -1)
  echo "\$MY_IP" > "$REMOTE_DIR/bound_ip.txt"
  export OLLAMA_HOST=\$MY_IP:$OLLAMA_PORT
  # Ollama sizes the KV cache as num_ctx * OLLAMA_NUM_PARALLEL, not just num_ctx --
  # unset, docs.ollama.com/faq says the default is 1, but the env var has been reported
  # ignored in favor of a hardcoded 4: https://github.com/ollama/ollama/issues/5722
  # At this NUM_CTX, 4x would be a guaranteed OOM on the L40S 48GB card before even loading
  # the weights. Pinned explicitly: this is a single-user coding-agent session, no
  # concurrent requests to batch anyway.
  export OLLAMA_NUM_PARALLEL=1
  ollama serve > "$REMOTE_DIR/ollama_serve.log" 2>&1 &
  SERVE_PID=\$!
  sleep 1
  if kill -0 \$SERVE_PID 2>/dev/null; then
    echo "ollama serve alive 1s after start (pid \$SERVE_PID)" >> "$REMOTE_DIR/trace.log"
  else
    echo "ollama serve ALREADY DEAD 1s after start (pid \$SERVE_PID) -- see ollama_serve.log" >> "$REMOTE_DIR/trace.log"
  fi
  until curl -sf "http://\$OLLAMA_HOST/api/tags" >/dev/null 2>&1; do
    if ! kill -0 \$SERVE_PID 2>/dev/null; then
      echo "ollama serve died while waiting on health check -- see ollama_serve.log" >> "$REMOTE_DIR/trace.log"
      break
    fi
    sleep 1
  done
  ollama pull $BASE_MODEL
  printf "FROM %s\nPARAMETER num_ctx %s\n" "$BASE_MODEL" "$NUM_CTX" > "$REMOTE_DIR/Modelfile"
  ollama create $CTX_MODEL -f "$REMOTE_DIR/Modelfile"
  wait \$SERVE_PID
  echo "wait \$SERVE_PID returned, serve exit code = \$?" >> "$REMOTE_DIR/trace.log"
'
LAUNCH

ssh_quiet "$REMOTE_HOST" "mkdir -p \"$REMOTE_DIR\""
scp -q "$LOCAL_LAUNCH_SCRIPT" "$REMOTE_HOST:$REMOTE_DIR/launch.sh"
rm -f "$LOCAL_LAUNCH_SCRIPT"

echo "==> Starting GPU allocation + Ollama server, detached on the Oscar login node..."
echo "    (progress/errors: ssh $REMOTE_HOST cat $REMOTE_DIR/launch.log)"
ssh_quiet "$REMOTE_HOST" "chmod +x \"$REMOTE_DIR/launch.sh\"; setsid nohup bash \"$REMOTE_DIR/launch.sh\" > \"$REMOTE_DIR/launch.log\" 2>&1 < /dev/null & disown; sleep 1; echo LAUNCHED"

echo "==> Waiting for the allocation to land (this also covers the model pull on first run)..."
NODE=""
for i in $(seq 1 90); do
  NODE=$(ssh_quiet "$REMOTE_HOST" "squeue -u \$(whoami) -n $JOB_NAME -t RUNNING -h -o '%N'" 2>/dev/null | head -1 || true)
  [ -n "$NODE" ] && [ "$NODE" != "(null)" ] && break
  sleep 2
done
if [ -z "$NODE" ] || [ "$NODE" = "(null)" ]; then
  echo "==> Attempt $((ATTEMPT_NUM + 1)) never landed (allocation didn't start within 3min). Check: cat $REMOTE_DIR/launch.log" >&2
  cleanup_failed_attempt
  continue
fi
echo "==> Allocated node: $NODE"

# Safety gate, non-negotiable: never let anything heavy (a model pull, ollama serve)
# proceed without first confirming it's actually executing on the allocated GPU node
# and not the shared login node. This is exactly the failure this script had earlier.
RAN_ON=""
for i in $(seq 1 30); do
  RAN_ON=$(ssh_quiet "$REMOTE_HOST" "cat \"$REMOTE_DIR/ran_on.txt\" 2>/dev/null" || true)
  [ -n "$RAN_ON" ] && break
  sleep 1
done
if [ -z "$RAN_ON" ]; then
  echo "==> Attempt $((ATTEMPT_NUM + 1)): job landed but never reported its hostname. Check: ssh $REMOTE_HOST cat $REMOTE_DIR/launch.log" >&2
  cleanup_failed_attempt
  continue
fi
if ! echo "$RAN_ON" | grep -q "$NODE"; then
  echo "SAFETY ABORT: the job reports running on '$RAN_ON', not the allocated node '$NODE'. Refusing to proceed -- this would mean running the model pull/server on the wrong machine (the login node)." >&2
  ssh_quiet "$REMOTE_HOST" "scancel --me --name=$JOB_NAME 2>/dev/null"
  exit 1
fi
echo "==> Confirmed running on the allocated node ($RAN_ON)."

# Read back the exact IP the compute node bound Ollama to, rather than independently
# re-resolving it ourselves. Found by testing: the login node resolving the node's
# hostname (or even ssh's own forwarding subprocess resolving it) doesn't reliably
# match what the compute node resolves for itself -- self-resolution on a multi-homed
# node isn't guaranteed to agree with how other hosts see it. Cheapest fix: have the
# compute node just tell us, directly, via a file on shared scratch.
NODE_IP=""
for i in $(seq 1 60); do
  NODE_IP=$(ssh_quiet "$REMOTE_HOST" "cat \"$REMOTE_DIR/bound_ip.txt\" 2>/dev/null" || true)
  [ -n "$NODE_IP" ] && break
  sleep 1
done
if [ -z "$NODE_IP" ]; then
  echo "==> Attempt $((ATTEMPT_NUM + 1)): job running but never reported its bound IP. Check: ssh $REMOTE_HOST cat $REMOTE_DIR/launch.log" >&2
  cleanup_failed_attempt
  continue
fi

echo "==> Opening tunnel localhost:$OLLAMA_PORT -> $NODE ($NODE_IP):$OLLAMA_PORT ..."
# Explicitly bind 127.0.0.1 and silence tunnel stderr to prevent macOS Darwin "setsockopt TCP_NODELAY: Invalid argument" noise
ssh -N -q -L "127.0.0.1:$OLLAMA_PORT:$NODE_IP:$OLLAMA_PORT" "$REMOTE_HOST" >/dev/null 2>&1 &
TUNNEL_PID=$!

echo "==> Waiting for $CTX_MODEL to be ready (first run on an uncached model can take several minutes)..."
READY=false
JOB_DIED=false
for i in $(seq 1 400); do
  # Plain substring match, not exact-quoted: Ollama typically registers created models
  # with a ":latest" suffix (e.g. "qwen3.8-27b-256k:latest"), which an exact quoted match
  # would miss entirely.
  if curl -sf "http://127.0.0.1:$OLLAMA_PORT/api/tags" 2>/dev/null | grep -q "$CTX_MODEL"; then
    READY=true
    break
  fi
  # Don't just sit out the full ~13min timeout if the allocation itself already died
  # (confirmed real failure mode: job gets revoked ~1-2s after landing, on a heavily
  # contested shared GPU pool -- checking every 2s here catches that in seconds instead).
  if ! ssh_quiet "$REMOTE_HOST" "squeue -u \$(whoami) -n $JOB_NAME -t RUNNING -h -o '%N'" 2>/dev/null | grep -q .; then
    JOB_DIED=true
    break
  fi
  sleep 2
done
if [ "$READY" = true ]; then
  echo "==> $CTX_MODEL is up and ready."
  SUCCESS=true
  break
fi
if [ "$JOB_DIED" = true ]; then
  echo "==> Attempt $((ATTEMPT_NUM + 1)): allocation died while waiting for the model (job no longer RUNNING)." >&2
else
  echo "==> Attempt $((ATTEMPT_NUM + 1)): $CTX_MODEL never became ready within ~13min, but the job is still running -- likely a real problem, not contention. Check: ssh $REMOTE_HOST cat $REMOTE_DIR/launch.log" >&2
fi
cleanup_failed_attempt
done

if [ "$SUCCESS" != true ]; then
  echo "" >&2
  echo "All $NUM_ATTEMPTS attempts failed. Tried: ${GRES_OPTIONS[*]:0:$NUM_ATTEMPTS}." >&2
  echo "This is very likely cluster-wide GPU contention on the shared pool, not a bug in this" >&2
  echo "script -- check current availability yourself: ssh $REMOTE_HOST sinfo -p gpu -N -o '%N %t %G'" >&2
  exit 1
fi

if [ "$BACKEND" = "opencode" ]; then
  CONFIG_DIR="$HOME/.config/opencode"
  CONFIG_FILE="$CONFIG_DIR/opencode.json"
  mkdir -p "$CONFIG_DIR"
  cat >"$CONFIG_FILE" <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama",
      "options": { "baseURL": "http://127.0.0.1:$OLLAMA_PORT/v1" },
      "models": { "$CTX_MODEL": { "name": "$CTX_MODEL" } }
    }
  }
}
EOF
  echo "==> Wrote $CONFIG_FILE"
  echo "==> Launching opencode. Exit it (or Ctrl-C this script) to tear everything down."
  opencode

elif [ "$BACKEND" = "claude" ]; then
  # Ollama exposes a native Anthropic-Messages-API-compatible endpoint at the server
  # root (no /v1 suffix, no separate proxy) -- confirmed via Ollama's own docs:
  # docs.ollama.com/integrations/claude-code. Requires a build at least as recent as
  # the one this script downloads; not available on Oscar's stale 0.21.0 module.
  echo "==> Configuring Claude Code for Ollama (native Anthropic Messages API endpoint)..."
  export ANTHROPIC_AUTH_TOKEN="ollama"
  export ANTHROPIC_API_KEY=""
  export ANTHROPIC_BASE_URL="http://127.0.0.1:$OLLAMA_PORT"
  # Inform Claude Code of the real context window and disable unknown-model fallback warning
  export CLAUDE_CODE_MAX_CONTEXT_TOKENS="$NUM_CTX"
  export CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT=1
  echo "==> Context window: $NUM_CTX tokens ($CTX_LABEL)."
  echo "==> Launching claude. Exit it (or Ctrl-C this script) to tear everything down."
  claude --model "$CTX_MODEL"
fi
