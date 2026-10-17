# CPT Training Pipeline

This directory contains the dataset preparation and online streaming training pipeline for **Cascade Parallel Thoughts (CPT)** adapters.

---

## 1. Multi-Domain Dataset Composition

The training dataset is generated via [`prepare_rich_dataset.py`](prepare_rich_dataset.py). It combines mathematical reasoning, competitive code generation, and low-level systems engineering:

| Domain / Source | Source Dataset | Size | Purpose |
| :--- | :--- | :--- | :--- |
| **Arithmetic & Step-by-Step CoT** | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) | 150 samples | Multi-step mathematical reasoning. |
| **Real-world Software Engineering** | [`princeton-nlp/SWE-bench_Lite`](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) | 150 samples | GitHub issue resolution, repository comprehension, and unified git patches. |
| **Olympiad Competition Math** | [`AI-MO/NuminaMath-CoT`](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | 250 samples | Mathematical proofs, number theory, and combinatorics. |
| **Advanced Mathematics** | [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | 150 samples | Algebra, calculus, geometry, and probability. |
| **Systems & Concurrency Engineering** | Curated Low-Level Corpus | Diverse | Lock-free Rust MPSC queues, cache-padded atomic indices, C++20 bump-pointer monolithic arenas, distributed transaction coordination (Saga vs 2PC). |

### Dataset Generation Command:
```sh
pip install -r requirements.txt
python prepare_rich_dataset.py rich_train_dataset.txt
```

---

## 2. In-Graph Latent Streaming Architecture (`train_pyramidal_cpt.py`)

CPT trains pyramidal adapters via direct on-the-fly streaming from the inference engine:

* **In-Graph Latent Streaming:** The inference engine (`llama-server` compiled with CPT hooks and launched via `--cascade-train-port 5555`) intercepts the active fork hidden states ($x$) and target layer states ($y$) directly in VRAM and streams them in real time through the TCP socket.
* **On-the-Fly Optimization:** `train_pyramidal_cpt.py` receives the stream of latents over the socket, computes the Smooth L1 loss on the residual delta ($\Delta h_{\text{residual}} = h_{\text{target}} - h_{\text{merge}}$), updates the adapter weights via AdamW, and frees tensor memory immediately after each step:
   $$\mathcal{L} = \text{SmoothL1}\left( \text{HubAdapter}(h_{\text{forks}}),\, \Delta h_{\text{residual}} \right)$$

### Training Command:
```sh
python train_pyramidal_cpt.py \
    --model /path/to/gemma-4-12B-it-QAT-Q4_0.gguf \
    --dataset rich_train_dataset.txt \
    --output-dir ../weights/cpt \
    --server-bin ../build/bin/Release/llama-server \
    --dim 3840 \
    --num-hubs 5 \
    --lr 1e-4 \
    --ngl 99
```

---

## 3. Output Artifacts

Upon completion, the trainer exports compact binary weights (~140 MB total) directly compatible with `llama-cli --cascade-dir`:
* `w_merge_1.bin` through `w_merge_5.bin` (FP16 merge weights)
* `gate_1.bin` through `gate_5.bin` (FP16 soft-gating vectors)
* `sin_scale.bin` (FP32 learned harmonic sine scale multipliers for each of the 14 forks)
