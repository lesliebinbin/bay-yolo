#!/usr/bin/env bash
# Start llama.cpp using the GGUF blobs and parameters of an Ollama model.
#
# Usage: ./llama-gemma4-31b.sh [model[:tag]] [extra llama-server args...]
#
# Requires `ollama` and `llama-server` in PATH. The model, vision projector,
# optional draft model, and sampling parameters are resolved dynamically with
# `ollama show --modelfile`. If no Ollama server is running, a temporary one
# is started just long enough to read the Modelfile, then stopped again.
set -euo pipefail

if [[ $# -gt 0 && "$1" != -* ]]; then
  MODEL_NAME="$1"
  shift
else
  MODEL_NAME="gemma4:31b"
fi

# --- temporary Ollama server bootstrap -----------------------------------
OLLAMA_ADDR="${OLLAMA_HOST:-127.0.0.1:11434}"
OLLAMA_ADDR="${OLLAMA_ADDR#http://}"
OLLAMA_ADDR="${OLLAMA_ADDR#https://}"
OLLAMA_URL="http://${OLLAMA_ADDR}"

OLLAMA_PID=""
OLLAMA_LOG=""

cleanup() {
  if [[ -n "$OLLAMA_PID" ]]; then
    kill "$OLLAMA_PID" 2>/dev/null || true
    wait "$OLLAMA_PID" 2>/dev/null || true
    OLLAMA_PID=""
  fi
  if [[ -n "$OLLAMA_LOG" ]]; then
    rm -f "$OLLAMA_LOG"
    OLLAMA_LOG=""
  fi
}
trap cleanup EXIT

is_up() { curl -sf "$OLLAMA_URL/api/tags" >/dev/null 2>&1; }

if ! is_up; then
  # Give an already-starting server a moment to come up.
  for _ in 1 2 3; do
    sleep 1
    is_up && break
  done
fi

if ! is_up; then
  echo "ollama server is not running; starting a temporary one..." >&2
  OLLAMA_LOG=$(mktemp)
  ollama serve >"$OLLAMA_LOG" 2>&1 &
  OLLAMA_PID=$!

  for _ in $(seq 1 30); do
    is_up && break
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
      echo "error: 'ollama serve' exited unexpectedly:" >&2
      cat "$OLLAMA_LOG" >&2
      exit 1
    fi
    sleep 1
  done

  if ! is_up; then
    echo "error: 'ollama serve' did not become ready at $OLLAMA_URL" >&2
    cat "$OLLAMA_LOG" >&2
    exit 1
  fi
fi

# --- resolve model, projector, draft model, and Ollama parameters --------
if ! MODELFILE=$(ollama show "$MODEL_NAME" --modelfile 2>/dev/null); then
  echo "error: could not read model '$MODEL_NAME'" >&2
  echo "       did you run 'ollama pull $MODEL_NAME'?" >&2
  exit 1
fi

MODEL=$(printf '%s\n' "$MODELFILE" | awk '/^FROM / {print $2; exit}')
MMPROJ=$(printf '%s\n' "$MODELFILE" | awk '/^FROM / {n++; if (n == 2) {print $2; exit}}')
DRAFT=$(printf '%s\n' "$MODELFILE" | awk '/^DRAFT / {print $2; exit}')

modelfile_param() {
  printf '%s\n' "$MODELFILE" | awk -v k="$1" '$1 == "PARAMETER" && $2 == k {print $3; exit}'
}

TEMP=$(modelfile_param temperature)
TOP_K=$(modelfile_param top_k)
TOP_P=$(modelfile_param top_p)
DRAFT_N_MAX=$(modelfile_param draft_num_predict)

[[ -n "$MODEL" ]]  || { echo "error: no model path in Modelfile for '$MODEL_NAME'" >&2; exit 1; }
[[ -n "$MMPROJ" ]] || { echo "error: no projector path (is '$MODEL_NAME' multimodal?)" >&2; exit 1; }

# Stop the temporary server now that the paths are known.
if [[ -n "$OLLAMA_PID" ]]; then
  echo "stopping temporary ollama server" >&2
  cleanup
fi
trap - EXIT

# --- launch llama-server -------------------------------------------------
# gemma4:31b -> gemma4-31b; user/model:tag -> user-model-tag
ALIAS="${MODEL_NAME//:/-}"
ALIAS="${ALIAS//\//-}"

ARGS=(
  -m "$MODEL"
  --mmproj "$MMPROJ"
  -c 262144
  -np 1
  -b 512
  -ub 512
  --context-shift
  --keep 4
  --jinja
  --no-webui
  --alias "$ALIAS"
)

# Match Ollama Modelfile speculative decoding + sampling parameters.
if [[ -n "$DRAFT" ]]; then
  ARGS+=(--spec-draft-model "$DRAFT")
  [[ " $* " == *" --spec-type "* ]] || ARGS+=(--spec-type draft-mtp)
  [[ -n "$DRAFT_N_MAX" ]] && ARGS+=(--spec-draft-n-max "$DRAFT_N_MAX")
fi
[[ -n "$TEMP" ]]  && ARGS+=(--temp "$TEMP")
[[ -n "$TOP_K" ]] && ARGS+=(--top-k "$TOP_K")
[[ -n "$TOP_P" ]] && ARGS+=(--top-p "$TOP_P")

# Only add defaults for host/port if the caller did not override them.
[[ " $* " == *" --host "* ]] || ARGS+=(--host 127.0.0.1)
[[ " $* " == *" --port "* ]] || ARGS+=(--port 11436)

exec llama-server "${ARGS[@]}" "$@"
