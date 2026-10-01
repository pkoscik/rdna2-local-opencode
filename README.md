# pkoscik's Local OpenCode setup

A local LLM inference setup using llama.cpp (TurboQuant fork) with ROCm on AMD hardware, serving as an OpenCode backend.

## Hardware

- **GPU:** AMD Radeon RX 6800 XT (16 GB, `gfx1030`, RDNA2)
- **CPU:** AMD Ryzen 7 7700X (`gfx1036` - hidden from llama.cpp)
- **RAM:** 64 GB system
- **OS:** Arch-based

## Quick Start

```bash
# 1. Build llama.cpp TurboQuant fork
./build.sh

# 2. Pick a config in the TUI, tweak knobs, launch (downloads the model on first use)
./run.sh

# 3. Point OpenCode to http://127.0.0.1:8080/v1
```

## Configs

`./run.sh [name]` loads `config/<name>.conf` (default `moe`). In a terminal it opens a small whiptail TUI: Enter edits a knob, `LOAD` switches to another config, `SAVE_AS` copies the knobs to a new config, and `LAUNCH` starts the server. Edits are written back to the loaded config file. With `NO_TUI=1` or no TTY it launches directly. Env vars override the config for one run.

| Config | Model | Context | KV | CPU-MoE | Spec | Best for |
|--------|-------|---------|----|---------|------|----------|
| `moe` | 35B-A3B Q6_K | 131k | f16 | 28 | MTP | programming agent + chat |
| `moe-long` | 35B-A3B Q6_K | 262k | f16 | 32 | MTP | whole-repo context |
| `dense` | 27B IQ4_XS | 32k | q8_0 | - | MTP | hard chat questions with thinking |

```bash
./run.sh                           # TUI on config/moe.conf
./run.sh dense                     # TUI on config/dense.conf
NO_TUI=1 ./run.sh moe-long         # launch directly
CTX=65536 NO_TUI=1 ./run.sh        # one-off override, not saved
THINKING=off ./run.sh              # faster agent loops
```

## Setup

### 1. Install dependencies

```bash
sudo pacman -Syu

# ROCm SDK
sudo pacman -S rocm-hip-sdk rocm-hip-runtime rocm-opencl-runtime \
               hipblas rocblas rocsolver rocsparse rocwmma

# Build dependencies
sudo pacman -S base-devel cmake ninja git curl

# GPU permissions
sudo usermod -aG video,render $USER
```

Reboot or re-login for group changes to take effect.

### 2. Verify ROCm

```bash
rocminfo | grep -E 'gfx|Name'
#   Name: gfx1030        - your RX 6800 XT
#   Name: gfx1036        - Ryzen iGPU (must be hidden at runtime)
```

### 3. Build the TurboQuant fork

```bash
./build.sh
```

This clones `https://github.com/TheTom/llama-cpp-turboquant`, checks out the `feature/turboquant-kv-cache` branch, and builds with ROCm.

> **`GGML_HIP_ROCWMMA_FATTN=OFF` is required for RDNA2** (the WMMA fast-attention path only exists on RDNA3+)

### 4. Configure OpenCode

```bash
sudo pacman -S opencode

./init_opencode.sh
```

The model ID (`qwen38`) is just a label - `llama-server` serves whatever GGUF is loaded. No config change is needed when switching modes; just stop the server, switch mode, restart, and start a fresh session in OpenCode.

## Models

`run.sh` downloads the selected GGUF on first use:

| Model | File | Size | Source | Best for |
|-------|------|------|--------|----------|
| Qwen3.8-35B-A3B Q6_K | `Qwen3.8-35B-A3B-Q6_K.gguf` | 29.2 GB | [empero-ai/Qwen3.8-35B-A3B-Distill-GGUF](https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill-GGUF) | Daily driver (`moe`, `moe-long`) |
| Qwen3.8-27B UD-IQ4_XS | `Qwen3.8-27B-UD-IQ4_XS.gguf` | 14.3 GB | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | Hard chat questions (`dense`) |

## Tuning Knobs

| Knob | What it does | Higher | Lower |
|------|-------------|--------|-------|
| `CTX` | Context window in tokens | Remembers more, slower, more VRAM | Snappier, less memory |
| `B` / `UB` | Batch / micro-batch for prompt eval | Faster ingest, more VRAM | Slower ingest, fits bigger contexts |
| `THINKING` | Internal reasoning before answering | Better one-shot quality, much slower | Faster, fine for agent loops |
| `THINK_BUDGET` | Max thinking tokens per turn | More deliberation | Avoids token spirals |
| `TEMP` / `TOP_P` / `PRESENCE` | Sampling (`auto` = model card values for the thinking mode) | More varied | More deterministic |
| `N_CPU_MOE` | MoE expert layers kept in RAM | Less VRAM, slower | Faster, OOM below the fit limit |
| `CTK` / `CTV` | KV cache precision (`f16`, `q8_0`, `turbo2/3/4`) | - | Quantized = less VRAM, slower decode at depth on RDNA2 |
| `CACHE_RAM` | Host-RAM prompt cache in MiB (0 = off) | More conversations survive side requests | - |
| `EXTRA` | Extra `llama-server` args, e.g. speculative decoding | - | - |
