# llama.cpp: Cascade Parallel Thoughts (CPT) & Brainstorm

<div align="center">

<b>In-Graph Latent Divergence and Gated Multi-Fork Fusion for LLM Inference</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[Architecture](#architecture) / [Telemetry & Benchmarks](#performance--hardware-telemetry) / [Speculative Decoding](#speculative-decoding--hardware-acceleration) / [Training Pipeline (training/)](training/README.md) / [CLI Reference](#cli-reference) / [Build & Run](#build-and-run) / [Citation](#citation)

</div>

---

> [!NOTE]
> **Note:** Code modifications, graph integration, and documentation were created by **LionPa** with AI pair-programming assistance.

---

## Technical Overview

Autoregressive models on complex tasks often rely on Chain-of-Thought (CoT) to structure reasoning sequentially. CPT complements token-level reasoning by adding latent-space exploration directly within the forward pass of the model.

This fork adds **Cascade Parallel Thoughts (CPT)** and **Brainstorm** directly into the `ggml` computation graph:
* **CPT:** Evaluates alternative latent trajectories across Transformer feed-forward layers using parallel branch forks and learned harmonic perturbations, merging them back into the residual stream with linear fusion matrices ($W_{\text{merge}}$) and soft-gating ($W_{\text{gate}}$).
* **Brainstorm:** Evaluates alternative outputs across parallel sequence slots (`-np`) in a single batch pass.

---

## Performance & Hardware Telemetry

Empirical measurements were taken on a single consumer GPU (**NVIDIA GeForce RTX 5060 Ti 16GB, 140W TDP**) running [Gemma 4 12B QAT Q4_0](https://huggingface.co/google/gemma-4-12b-it):

| Configuration | Generation Speed | Latency Overhead | VRAM Usage | Extra Disk Footprint |
| :--- | :--- | :--- | :--- | :--- |
| **Stock `llama.cpp`** | **108.2 tok/s** | Baseline (0%) | 7.42 GB | 0 MB |
| **`llama.cpp` + CPT** | **94.1 tok/s** | **~13.0%** | 7.56 GB | **+140 MB** |

### Interaction with Chain-of-Thought (CoT):
CPT does not replace token-level Chain-of-Thought; it operates alongside it:
* **Token-Level CoT:** Expands reasoning sequentially by generating intermediate tokens into the context window.
* **In-Graph CPT:** Enriches representations within each forward step across intermediate FFN layers at a constant ~13% latency cost.
* When evaluating models with thinking tokens, CPT operates on every forward step without modifying prompt templates, token emission, or context contracts.

---

## Speculative Decoding & Hardware Acceleration

Because CPT operates within the forward graph without modifying KV-cache layout or output vocabulary logits, it is compatible with Speculative Decoding and Flash Attention:

* **Multi-Token Prediction (MTP) Draft Assistant:** When paired with a draft model (e.g. `gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf`) via `--spec-type draft-mtp --spec-draft-n-max 4`, accepted draft tokens offset the latency of the additional branch computation.
* **Flash Attention (`-fa on`):** Enables efficient $O(N)$ memory attention scaling for large context windows (tested up to 100k context).

---

## Architecture

CPT does not duplicate base model weights. It introduces a compact set of linear fusion adapters ($d_{\text{model}} \times d_{\text{model}}$) and gating parameters (**140 MB total**) directly mapped into the `ggml` graph:

### 1. Incubation Forks (Layers 18–31)
Across layers 18 to 31, intermediate states $h_l$ are branched into 14 parallel paths that run through the layer's feed-forward network without altering the attention KV cache.

### 2. Harmonic Perturbation Injection ($\sin$-steering)
To introduce variance between branches, a sinusoidal perturbation scaled by learned fork parameters $s_f$ (from `sin_scale.bin`) is added to the hidden states:
$$\tilde{h}_l = h_l + \alpha \cdot |s_f| \cdot \sin(h_l)$$
where $\alpha$ is set at runtime via `--cascade-sin`.

### 3. Pyramidal Gated Fusion Hubs (Layers 21, 24, 27, 30, 33)
Branch states are aggregated at five merge layers. Each hub normalizes the incoming branch states, projects them through linear matrix $W_{\text{merge}}$, and modulates the update with a sigmoid gate:
$$x_{\text{hub}} = \frac{1}{\sqrt{|\text{Hub}|}} \sum_{f \in \text{Hub}} \text{RMSNorm}(h_f)$$
$$\Delta h = (W_{\text{merge}} \cdot x_{\text{hub}}) \odot \sigma(W_{\text{gate}} \cdot x_{\text{hub}})$$
$$h_{\text{out}} = h_{\text{trunk}} + \Delta h$$

---

## Brainstorm: Parallel Sampling

Brainstorm uses parallel sequence slots (`-np`) in `llama-cli` to sample multiple responses in a single run:
1. **Parallel Execution (`-np`):** Runs multiple sequences in parallel on the GPU with shared model weights.
2. **Steered Sampling:** When combined with `--cascade-sin`, each slot samples from the perturbed latent state.
3. **Independent Context:** Each sequence maintains its own KV cache while sharing the compute graph.

---

## Build and Run

### Compilation
Build with CUDA acceleration using standard CMake:

```sh
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

---

### Setup: Models and CPT Weights

1. **Base Model:** Download [Gemma 4 12B it QAT Q4_0](https://huggingface.co/lmstudio-community/gemma-4-12B-it-QAT-GGUF) into `./models/`.
2. **Draft Assistant (MTP):** Download [Gemma 4 12B MTP Q8_0 Assistant](https://huggingface.co/Janvitos/gemma-4-12B-it-qat-assistant-MTP-Q8_0-GGUF) into `./models/`.
3. **CPT Weights:** Download `gemma-4-12b-qat-cpt-weights.zip` from [GitHub Releases](https://github.com/LionPa/llama.cpp/releases) and extract to `./cpt_weights/` (contains `w_merge_*.bin`, `gate_*.bin`, `sin_scale.bin`).

---

### Server Mode (`llama-server`)

The `-np` flag sets the number of parallel sequence slots (batches). It must be equal to or greater than the number of candidates you plan to run simultaneously:

**Windows (PowerShell):**
```powershell
& ".\build\bin\Release\llama-server.exe" `
    -m "./models/gemma-4-12B-it-QAT-Q4_0.gguf" `
    --model-draft "./models/gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf" `
    --spec-type draft-mtp `
    --spec-draft-n-max 4 `
    -ngl 99 -fa on -c 100000 `
    --cascade-dir "./cpt_weights" `
    --cascade-sin 1.0 `
    --repeat-penalty 1.15 `
    -np 4 --port 8080 --temp 0.2
```

**Linux (Bash):**
```sh
./build/bin/Release/llama-server \
    -m ./models/gemma-4-12B-it-QAT-Q4_0.gguf \
    --model-draft ./models/gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf \
    --spec-type draft-mtp \
    --spec-draft-n-max 4 \
    -ngl 99 -fa on -c 100000 \
    --cascade-dir ./cpt_weights \
    --cascade-sin 1.0 \
    --repeat-penalty 1.15 \
    -np 4 --port 8080 --temp 0.2
```

---

### Interactive CLI with Adversarial Brainstorm

To use Brainstorm in interactive chat, set `-np` to at least the number of candidates (e.g. `-np 3` for a 3-way brainstorm):

**Windows (PowerShell):**
```powershell
& ".\build\bin\Release\llama-cli.exe" `
    -m "./models/gemma-4-12B-it-QAT-Q4_0.gguf" `
    --model-draft "./models/gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf" `
    --spec-type draft-mtp `
    --spec-draft-n-max 4 `
    -ngl 99 -fa on -c 100000 `
    --cascade-dir "./cpt_weights" `
    --cascade-sin 1.0 `
    --repeat-penalty 1.1 `
    -np 3 --temp 0.2
```

**Linux (Bash):**
```sh
./build/bin/Release/llama-cli \
    -m ./models/gemma-4-12B-it-QAT-Q4_0.gguf \
    --model-draft ./models/gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf \
    --spec-type draft-mtp \
    --spec-draft-n-max 4 \
    -ngl 99 -fa on -c 100000 \
    --cascade-dir ./cpt_weights \
    --cascade-sin 1.0 \
    --repeat-penalty 1.1 \
    -np 3 --temp 0.2
```

In the interactive prompt, paste the `/brainstorm` command as a single line:

```text
/brainstorm 3, 1.0, 4.6, 23, "Review my original request and the 3 distinct candidates developed below:\n\n[Candidate 1 | Conventional & Grounded]:\n%1\n\n[Candidate 2 | Balanced & Adaptive]:\n%2\n\n[Candidate 3 | Lateral & Unconventional]:\n%3\n\nCONTEXT & NATURE OF CANDIDATES:\nThe three candidates above explore the problem from different analytical angles:\n- Candidate 1: Conventional and strictly grounded in core constraints.\n- Candidate 2: Balanced, capturing nuanced interactions and trade-offs.\n- Candidate 3: Lateral perspective, exploring non-obvious mechanisms and edge cases.\n\nCRITICAL INSTRUCTIONS:\n1. ROLE & HUMILITY (NO SELF-SOLVING): You are an Adversarial Code Reviewer and Lead Editor. You are NOT the author and you are NOT smarter than the collective candidates. DO NOT attempt to solve the problem from scratch, do not invent new architectures, and do not brainstorm alternatives. The correct solution already exists across these candidates.\n2. ADVERSARIAL AUDIT (NO CANDIDATE IS PERFECT): Never assume any single candidate is 100% flawless. Assume each candidate has at least one subtle bug (unhandled edge cases, off-by-one errors, dimensional mismatches, or missing syntax/imports). Cross-examine the candidates against each other to expose and eliminate these silent errors.\n3. SELECT THE PRIMARY BACKBONE (85-95%): Identify which Candidate provides the most complete, concrete, and production-ready solution (explicit formulas, fully written code without placeholders/TODOs). Adopt this candidate as your single architectural backbone.\n4. SURGICAL PATCHING: Patch the backbone locally using specific bug-fixes and edge-case handling from secondary candidates. Screen Candidate 3 for clever algorithmic optimizations or rare safeguards, and cleanly graft them into the backbone.\n5. NO INVENTED ENTITIES: Never introduce arbitrary variables, external libraries, or unrequested assumptions not present in the prompt or candidates.\n6. FINAL OUTPUT: Produce a seamless, complete, publication-ready response. Output ONLY the deliverable without meta-talk, introductions, or mentions of 'candidates', 'reviews', or the synthesis process.", 1.0
```

---

## CLI Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--cascade-dir` | `std::string` | `""` | Path to directory containing binary adapter weights (`w_merge_X.bin`, `gate_X.bin`, `sin_scale.bin`). |
| `--cascade-sin` | `float` | `1.0` | Scalar multiplier $\alpha$ for the non-linear harmonic perturbation vectors. |
| `--cpt-train-port` | `int` | `0` | Optional TCP streaming port for online trainer telemetry and latent state inspection. |
| `--model-draft` | `std::string` | `""` | Path to assistant draft model for speculative decoding (e.g. MTP draft GGUF). |
| `--spec-type` | `std::string` | `none` | Speculative decoding algorithm type (`draft-mtp`, `ngram-mod`, etc.). |
| `--spec-draft-n-max` | `int` | `16` | Maximum number of draft tokens proposed per step by the speculative assistant. |
| `-np`, `--parallel` | `int` | `1` | Number of parallel sequence slots for concurrent Brainstorm streams. |
| `-fa`, `--flash-attn` | `on\|off` | `off` | Enables Flash Attention kernel for long-context memory scaling. |

---

## Training CPT Adapters

To train custom CPT fusion adapters and learned gating hubs for any model, see the reproducible scripts in [`training/`](training/):
* **Dataset Generation:** [`training/prepare_rich_dataset.py`](training/prepare_rich_dataset.py) compiles reasoning problems from GSM8K, SWE-bench Lite, NuminaMath-CoT, MATH-500, and low-level systems engineering challenges.
* **In-Graph Streaming Trainer:** [`training/train_pyramidal_cpt.py`](training/train_pyramidal_cpt.py) connects directly to `llama-server` via TCP socket, streaming intermediate hidden state representations in real time to optimize the 5 pyramidal gating hubs.

Full setup instructions are detailed in [`training/README.md`](training/README.md).

---

## Citation

```bibtex
@misc{cpt2026,
  author = {LionPa},
  title = {Cascade Parallel Thoughts (CPT): In-Graph Latent Divergence and Gated Multi-Fork Fusion in llama.cpp},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/LionPa/llama.cpp}}
}
```

## Upstream Project & License
This project is an extension of [llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov and contributors.
Licensed under the **MIT License**. Original copyright (c) 2023-2026 The ggml authors. CPT extensions copyright (c) 2026 LionPa.
