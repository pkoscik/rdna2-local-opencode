#!/usr/bin/env bash
# Start llama-server with config/<name>.conf (default: moe). Env vars override the config:
#   CTX=65536 NO_TUI=1 ./run.sh dense
set -euo pipefail
cd "$(dirname "$0")"
shopt -s nullglob

BIN=./llama-cpp-turboquant/build/bin/llama-server
M27=models/Qwen3.8-27B-UD-IQ4_XS.gguf
M35=models/Qwen3.8-35B-A3B-Q6_K.gguf
declare -A URL=(
  [$M27]=https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-IQ4_XS.gguf
  [$M35]=https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill-GGUF/resolve/main/Qwen3.8-35B-A3B-Q6_K.gguf
)
KNOBS=(MODEL CTX THINKING THINK_BUDGET N_CPU_MOE B UB CTK CTV CACHE_RAM TEMP TOP_P PRESENCE PORT EXTRA)

die() { echo "✗ $*" >&2; exit 1; }

# Knobs already set in the environment win over every config.
declare -A ENV=()
for k in "${KNOBS[@]}"; do [[ -n ${!k:-} ]] && ENV[$k]=${!k}; done

defaults() {
  MODEL=$M35 CTX=131072 THINKING=on THINK_BUDGET=4096 N_CPU_MOE=28 B=2048 UB=2048
  CTK=f16 CTV=f16 CACHE_RAM=16384 TEMP=auto TOP_P=auto PRESENCE=auto PORT=8080
  EXTRA="--spec-type draft-mtp --spec-draft-n-max 3"
}
load() {
  NAME=$1 CONF=config/$1.conf
  defaults
  [[ -f $CONF ]] && source "$CONF"
  for k in "${!ENV[@]}"; do printf -v "$k" %s "${ENV[$k]}"; done
}
dump() { for k in "${KNOBS[@]}"; do printf '%s=%q\n' "$k" "${!k}"; done; }

wt() { whiptail --title "llama-server [$NAME]" "$@" 3>&1 1>&2 2>&3; }

tui() {
  local choice=LAUNCH val f items opts
  while :; do
    items=(LAUNCH "start server" LOAD "switch to another config/*.conf" SAVE_AS "copy the knobs to a new config")
    for k in "${KNOBS[@]}"; do items+=("$k" "${!k}"); done
    choice=$(wt --default-item "$choice" --cancel-button Quit \
      --menu "Enter edits a knob, changes are saved to $CONF (auto = model-recommended)" 0 0 0 -- "${items[@]}") || exit 1
    case $choice in
      LAUNCH) return ;;
      LOAD)
        opts=()
        for f in config/*.conf; do f=${f#config/}; opts+=("${f%.conf}" ""); done
        (( ${#opts[@]} )) || continue
        val=$(wt --default-item "$NAME" --menu "Config" 0 0 0 -- "${opts[@]}") && load "$val"
        continue ;;
      SAVE_AS) val=$(wt --inputbox "New config name" 0 60 -- "$NAME") && [[ -n $val ]] && NAME=$val CONF=config/$val.conf ;;
      THINKING) [[ $THINKING == on ]] && THINKING=off || THINKING=on ;;
      MODEL)
        opts=()
        for f in $(printf '%s\n' "$M35" "$M27" models/*.gguf | awk '!s[$0]++'); do
          opts+=("$f" "$([[ -f $f ]] && echo "✓ local" || echo "↓ will download")")
        done
        val=$(wt --default-item "$MODEL" --menu "Model" 0 0 0 -- "${opts[@]}") && MODEL=$val ;;
      *) val=$(wt --inputbox "$choice" 0 60 -- "${!choice}") && printf -v "$choice" %s "$val" ;;
    esac
    mkdir -p config && dump > "$CONF"
  done
}

load "${1:-${CONFIG:-moe}}"
[[ $# -gt 0 && ! -f $CONF ]] && die "no such config: $CONF (configs: $(ls config 2>/dev/null | sed 's/\.conf//' | xargs))"
if [[ -t 0 && -t 1 && -z ${NO_TUI:-} ]] && command -v whiptail >/dev/null; then
  tui
fi

# Sampling: Qwen3.8-27B card says temp 1.0 when thinking, the 35B one says 0.6.
if [[ $THINKING == on ]]; then
  [[ $TEMP == auto ]] && TEMP=$([[ $MODEL == *35B* ]] && echo 0.6 || echo 1.0)
  [[ $TOP_P == auto ]] && TOP_P=0.95
  [[ $PRESENCE == auto ]] && PRESENCE=0.0
  REASONING=(--reasoning on --reasoning-budget "$THINK_BUDGET")
else
  [[ $TEMP == auto ]] && TEMP=0.7
  [[ $TOP_P == auto ]] && TOP_P=0.8
  [[ $PRESENCE == auto ]] && PRESENCE=1.5
  REASONING=(--reasoning off)
fi
MOE=(); [[ $N_CPU_MOE -gt 0 ]] && MOE=(--n-cpu-moe "$N_CPU_MOE")
read -ra EXTRA_ARGS <<< "$EXTRA"

[[ -x $BIN ]] || die "llama-server not found at $BIN - run ./build.sh first"
! ss -tln 2>/dev/null | grep -q ":$PORT " || die "port $PORT in use (lsof -i :$PORT, or PORT=8081)"

if [[ ! -f $MODEL ]]; then
  [[ -v URL[$MODEL] ]] || die "model not found: $MODEL"
  echo "↓ downloading $(basename "$MODEL") ..."
  # .part so an interrupted download is resumed, never mistaken for a complete file
  wget --continue --show-progress -O "$MODEL.part" "${URL[$MODEL]}"
  mv "$MODEL.part" "$MODEL"
fi

echo; echo "  config: $CONF"; dump | sed 's/^/  /'; echo

# Hide the Ryzen iGPU from ROCm, pin the dGPU to its high-performance state
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0
for f in /sys/class/drm/card*/device/power_dpm_force_performance_level; do
  d=${f%/*}
  [[ $(<"$f") == high ]] || echo high | sudo tee "$f" >/dev/null || true
  [[ $(<"$d/power/control") == on ]] || echo on | sudo tee "$d/power/control" >/dev/null || true
done

exec "$BIN" \
  -m "$MODEL" --alias "$(basename "$MODEL" .gguf)" \
  --host 127.0.0.1 --port "$PORT" \
  -c "$CTX" -b "$B" -ub "$UB" -ngl 99 -fa 1 \
  --cache-type-k "$CTK" --cache-type-v "$CTV" \
  --cache-ram "$CACHE_RAM" --no-context-shift \
  --jinja "${REASONING[@]}" "${MOE[@]}" \
  --temp "$TEMP" --top-p "$TOP_P" --top-k 20 --min-p 0.0 \
  --presence-penalty "$PRESENCE" \
  -np 1 "${EXTRA_ARGS[@]}"
