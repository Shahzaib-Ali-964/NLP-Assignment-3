# CS-4063 NLP Assignment 3 — From-Scratch RAG Pipeline

**Student:** 22i-0576  
**Course:** CS-4063 Natural Language Processing  
**Semester:** Fall 2024 — Semester 7

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

A fully hand-written, from-scratch PyTorch implementation of a Retrieval-Augmented Generation (RAG) pipeline for Amazon product review analysis. The system combines a joint-classification encoder, a cosine-similarity vector index, and a causal decoder-only text generator — all built without any pre-trained modules (no `nn.Transformer`, no `transformers`, no `BertModel`).

---

## Table of Contents
- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Repository Layout](#repository-layout)
- [System Components](#system-components)
- [Environment Setup](#environment-setup)
- [Running the Project](#running-the-project)
- [Ablation Results](#ablation-results)

---

## Overview

This assignment implements a complete NLP pipeline across seven sequential phases:

| Phase | Description |
|-------|-------------|
| 0 | Runtime initialization — global `PARAMS`, seed fixing, directory creation |
| 1 | Dataset ingestion — streaming 36,000 Amazon reviews; stratified CSV splits |
| 2 | Encoder architecture — custom `AttentionLayer`, `TransformerEncoderLayer`, `SinusoidalPE` |
| 3 | Encoder training — joint polarity + category loss; AdamW + ReduceLROnPlateau |
| 4 | Vector retrieval — L2-normalized corpus index with cosine k-NN search |
| 5 | Decoder architecture — causal `TextGenerator` with autoregressive masking |
| 6 | Decoder training — teacher-forcing on RAG-augmented prompts |
| 7 | Evaluation — perplexity comparison (RAG vs. no-RAG), sample generation |

---

## Pipeline Architecture

```mermaid
graph TD
    A[Amazon Reviews HuggingFace Stream] -->|fetch_category_reviews| B(Preprocessing & Stratified Split)
    B -->|Lexicon build + ReviewCorpus| C[Data/train.csv · val.csv · test.csv]

    C --> D[JointClassificationModel - TextEncoder backbone]
    D -->|Polarity + Category CE loss| E[checkpoints/encoder.pt]
    D -->|[BOS] pooled repr_vec| F[VectorIndex - L2-Normalized]

    F -->|find_nearest cosine k-NN| G[Retrieved Context Docs]

    C --> H[TextGenerator - Decoder-Only]
    G -->|construct_generation_prompt| H
    H -->|autoregressive_decode| I[checkpoints/decoder.pt]

    I --> J((Gradio Web UI — app.py))
```

---

## Repository Layout

```text
NLP_Assign_03/
├── i220576-NLP-Assignment3.ipynb     # Main notebook (roll number 22i-0576)
├── README.md                         # This file
├── requirements.txt                  # Pinned dependency list
└── implementation/
    ├── app.py                        # Gradio interactive demo
    ├── build_final.py                # End-to-end pipeline rebuild script
    ├── generate_report_v2.py         # Auto-generates Report.docx
    ├── Data/
    │   ├── train.csv                 # 70% stratified training split  (25,200 rows)
    │   ├── val.csv                   # 15% validation split           ( 5,400 rows)
    │   └── test.csv                  # 15% held-out test split        ( 5,400 rows)
    ├── models/
    │   ├── encoder.pt                # Saved JointClassificationModel weights
    │   └── decoder.pt                # Saved TextGenerator weights
    └── results/
        ├── encoder_loss.png          # Training vs. validation loss curve
        ├── hyperparam_log.csv        # Per-run metric log
        ├── train_embeddings.pt       # Serialized L2-normalized corpus vectors
        └── train_metadata.pt         # Metadata mapping (texts, labels, indices)
```

---

## System Components

### 1. Joint Classification Encoder (`JointClassificationModel`)
A stack of custom `TransformerEncoderLayer` blocks with sinusoidal positional encodings (`SinusoidalPE`). The `[BOS]` token acts as a CLS-style aggregate representation. Two independent linear heads predict:
- **Polarity** (Negative / Neutral / Positive)
- **Product Category** (Electronics / Books / Clothing)

Combined training objective:

$$\mathcal{L} = \lambda_s \cdot \mathcal{L}_{\text{sentiment}} + \lambda_c \cdot \mathcal{L}_{\text{category}} = 1.0 \cdot CE_{\text{sent}} + 0.5 \cdot CE_{\text{cat}}$$

### 2. Vector Retrieval Index (`VectorIndex`)
Corpus embeddings are L2-normalized at load time so that inner-product search is equivalent to cosine similarity:

$$\text{sim}(q, d) = \frac{q \cdot d}{\|q\| \|d\|} = \hat{q} \cdot \hat{d}$$

`find_nearest()` calls `torch.topk` on the dot-product scores — no external indexing library required.

### 3. Causal Text Generator (`TextGenerator`)
A decoder-only transformer using an upper-triangular boolean mask to enforce autoregressive ordering. Retrieved context passages are prepended to the generation prefix (Prefix-LM style). Inference uses greedy decoding via `autoregressive_decode()`.

---

## Environment Setup

Requires **Python 3.10+**.

1. Create and activate a virtual environment:
    ```bash
    python -m venv nlp_env
    .\nlp_env\Scripts\activate   # Windows
    source nlp_env/bin/activate  # Linux / macOS
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    > `transformers` is deliberately excluded — all model code is written from scratch.

---

## Running the Project

### Interactive Demo (Gradio UI)
```bash
cd implementation
python app.py
```
Launches at `http://127.0.0.1:7860`. Enter any product review to get predicted sentiment, category, retrieved neighbors, and a generated rationale.

### Full Rebuild from Scratch
```bash
cd implementation
python build_final.py
```
Re-streams data from HuggingFace, retrains both models, re-extracts embeddings, and regenerates the notebook output.

> **Note:** Full CPU training (2 encoder epochs + 1 decoder epoch) takes roughly 10–20 minutes depending on hardware.

---

## Ablation Results

Perplexity (PPL) measures how confidently the decoder assigns probability to held-out tokens — lower is better.

| Configuration | Polarity Accuracy | Category Accuracy | Decoder PPL |
|---------------|:-----------------:|:-----------------:|:-----------:|
| No RAG (baseline) | 85.07% | 86.48% | ~50.2 |
| **Full RAG** | **85.07%** | **86.48%** | **~12.4** |

Retrieval context provides a ~4× reduction in perplexity, confirming that nearest-neighbor passages significantly constrain the decoder's token distribution without changing classification accuracy (which depends solely on the encoder).
