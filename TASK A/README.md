# TASK A — Multi-Turn Conversational Retrieval
## MT-RAG SemEval 2026 — IIMAS-UNAM

> **Paper:** *Hybrid Sparse-Dense Retrieval for Multi-Turn Conversational Search*  
> **Team:** IIMAS-RAG (Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas, UNAM)  
> **Task:** SemEval 2026 Task A — Passage Retrieval for Multi-Turn Conversations

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Data and Indices (Hugging Face)](#data-and-indices-hugging-face)
4. [Quick Start — Docker](#quick-start--docker)
5. [Quick Start — Local (Python)](#quick-start--local-python)
6. [Experiments](#experiments)
   - [0-baselines](#0-baselines)
   - [02-hybrid](#02-hybrid)
7. [Statistical Validation](#statistical-validation)
8. [Reproducing Results](#reproducing-results)
9. [API Keys Required](#api-keys-required)
10. [Citation](#citation)

---

## Overview

This folder contains the complete, reproducible pipeline for **Task A (Retrieval)** of MT-RAG SemEval 2026. The system retrieves relevant passages given multi-turn conversational queries across four specialized domains:

| Domain | Topic | # Conversations | # Passages |
|--------|-------|-----------------|------------|
| **CLAPNQ** | Wikipedia QA | ~1 500 | ~500 000 |
| **Cloud** | Tech documentation | ~1 200 | ~60 000 |
| **FiQA** | Financial QA | ~1 200 | ~57 000 |
| **Govt** | Government (FDA/EPA) | ~1 200 | ~90 000 |

### Key results (nDCG@10 on dev set)

| System | CLAPNQ | Cloud | FiQA | Govt | Avg |
|--------|--------|-------|------|------|-----|
| BM25 baseline (full history) | 0.261 | 0.313 | 0.292 | 0.275 | 0.285 |
| Voyage-3 baseline (oracle rewrite) | 0.521 | 0.549 | 0.534 | 0.511 | 0.529 |
| **Hybrid SPLADE + Voyage (RRF, oracle rewrite)** | **0.578** | **0.597** | **0.572** | **0.561** | **0.577** |

---

## Directory Structure

```
TASK A/
├── README.md                         ← This file
├── Dockerfile                        ← Reproducible CUDA 12.1 image
├── docker-compose.yml                ← Docker Compose with GPU + volume mounts
├── requirements.txt                  ← Python dependencies (pinned)
├── run.sh                            ← Main entry-point shell script
├── .env.example                      ← Template for API keys
│
├── configs/
│   ├── base.yaml                     ← Global defaults (seed, paths, compute)
│   ├── domains/
│   │   ├── clapnq.yaml               ← Domain-specific overrides
│   │   ├── cloud.yaml
│   │   ├── fiqa.yaml
│   │   └── govt.yaml
│   └── experiments/
│       ├── 0-baselines/              ← 14 baseline experiment configs
│       │   ├── A0_baseline_bm25_fullhist.yaml
│       │   ├── A0_baseline_splade_fullhist.yaml
│       │   ├── A1_baseline_bgem3_fullhist.yaml
│       │   ├── A1_baseline_voyage_fullhist.yaml
│       │   ├── A2_baseline_bge15_rewrite.yaml
│       │   ├── A2_baseline_bgem3_rewrite.yaml
│       │   ├── A2_baseline_bm25_rewrite.yaml
│       │   ├── A2_baseline_splade_rewrite.yaml
│       │   ├── A2_baseline_voyage_rewrite.yaml
│       │   └── replication_*.yaml    ← (5 replication baselines)
│       └── 02-hybrid/                ← 12 hybrid experiment configs
│           ├── hybrid_splade_bge15_norewrite.yaml
│           ├── hybrid_splade_bge15_rewrite.yaml
│           ├── hybrid_splade_voyage_norewrite.yaml
│           ├── hybrid_splade_voyage_rewrite.yaml
│           └── ...                   ← (variants with own rewrites, v2, v3, etc.)
│
├── src/
│   └── pipeline/
│       ├── run.py                    ← Main pipeline orchestrator
│       ├── retrieval/
│       │   ├── sparse.py             ← BM25 and SPLADE retrieval
│       │   ├── dense.py              ← BGE / Cohere dense retrieval
│       │   ├── voyage.py             ← Voyage-3 dense retrieval (API)
│       │   ├── hybrid.py             ← HybridRetriever (sparse + dense)
│       │   ├── fusion.py             ← RRF and linear fusion
│       │   └── analysis.py           ← Bootstrap CI, Wilcoxon, Bonferroni
│       ├── query_transform/
│       │   ├── rewriters.py          ← Cohere / vLLM query rewriters
│       │   └── expansion.py          ← HyDE, multi-query
│       ├── reranking/
│       │   ├── bge_reranker.py       ← BGE cross-encoder reranker
│       │   └── cohere_rerank.py      ← Cohere rerank API
│       └── evaluation/
│           ├── run_retrieval_eval.py ← nDCG@10, MRR@10, Recall computation
│           └── format_checker.py     ← Output format validation
│
├── scripts/
│   ├── run_experiment.py             ← CLI runner (single or all experiments)
│   ├── run_ablation_statistical_tests.py  ← Generates ablation_statistical_*.json
│   ├── legacy_statistical_validation.py   ← Full Wilcoxon + Bootstrap + Holm
│   └── run_all_analyses.py           ← Orchestrates all analyses
│
└── statistical_results/
    ├── statistical_summary_for_paper.txt   ← Human-readable summary (baselines + hybrid)
    ├── statistical_validation_report.json  ← Full validation (Wilcoxon + Bootstrap CI)
    ├── statistical_report.json             ← Concise statistical report
    ├── ablation_statistical_summary.txt    ← Ablation study summary
    └── ablation_statistical_tests.json     ← Detailed ablation tests (137 KB)
```

> **Heavy data (indices, corpora, submissions)** are hosted on Hugging Face — see section below.

---

## Data and Indices (Hugging Face)

All heavy artifacts are hosted at:

**🤗 [`vania-janet/mt-rag-benchmark-data`](https://huggingface.co/datasets/vania-janet/mt-rag-benchmark-data)**

| HF path | Content | Size |
|---------|---------|------|
| `indices/{dataset}/{model}/` | Pre-built FAISS / BM25 / SPLADE indices | ~6.8 GB |
| `data/passage_level_processed/{dataset}/corpus.jsonl` | Passage-level corpora | ~428 MB |
| `data/retrieval_tasks/{dataset}/` | Queries, qrels, rewritten queries | ~2.8 MB |
| `data/rewrites/cohere_v3/` | Final Cohere Command-R v3 rewrites | — |
| `data/rewrites/own_*/` | Own-trained rewriter outputs | — |
| `data/rewrites/hyde/` | HyDE hypothetical documents | — |
| `data/submissions/` | Retrieval results per experiment | ~1.9 GB |
| `statistical_tests/results/` | All statistical test outputs | ~5.8 MB |

### Downloading data

```bash
# Install HF Hub
pip install huggingface_hub

# Download everything
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='vania-janet/mt-rag-benchmark-data',
    repo_type='dataset',
    local_dir='./hf_data'
)
"

# Or download a specific folder (e.g., only retrieval tasks)
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='vania-janet/mt-rag-benchmark-data',
    repo_type='dataset',
    local_dir='./hf_data',
    allow_patterns='data/retrieval_tasks/*'
)
"
```

Then symlink/move the folders so they match the expected structure:

```bash
ln -s ./hf_data/indices          ./indices
ln -s ./hf_data/data             ./data
```

---

## Quick Start — Docker

This is the **recommended** way to run experiments — fully reproducible, CUDA 12.1.

### Prerequisites

- Docker ≥ 24.0 with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- GPU with ≥ 16 GB VRAM (for BGE-M3 / Voyage; BM25/SPLADE work on CPU)

### 1. Configure API keys

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

### 2. Download data from Hugging Face

```bash
# Place data/ and indices/ directories alongside this folder
# (or symlink them as shown above)
```

### 3. Build and start container

```bash
docker compose up -d --build
docker compose exec mtrag-retrieval bash
```

### 4. Run an experiment inside the container

```bash
# Single experiment, all domains
./run.sh -e hybrid_splade_voyage_rewrite -d all

# Run all baselines across all domains
./run.sh --category baselines -d all

# Run all hybrid experiments
./run.sh --category hybrid -d all

# Dry run (validate configs, no execution)
./run.sh -e A2_baseline_voyage_rewrite -d clapnq --dry-run
```

---

## Quick Start — Local (Python)

### Prerequisites

- Python 3.10 or 3.11
- pip

### 1. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

```bash
cp .env.example .env
source .env   # or export variables manually
export PYTHONPATH=.:src
```

### 4. Download data

See the [Data and Indices](#data-and-indices-hugging-face) section above.

### 5. Run an experiment

```bash
# Single experiment, single domain
python scripts/run_experiment.py \
    --experiment hybrid_splade_voyage_rewrite \
    --domain clapnq

# Single experiment, all domains
python scripts/run_experiment.py \
    --experiment A2_baseline_voyage_rewrite \
    --domain all

# Run all baseline experiments on all domains
python scripts/run_experiment.py \
    --experiment all \
    --domain all \
    --category baselines
```

Results are written to `data/submissions/{experiment_name}/{domain}/`:
- `retrieval_results.jsonl` — Retrieved passages per query
- `metrics.json` — nDCG@10, MRR@10, Recall@100, MAP@10

---

## Experiments

### 0-baselines

Located in `configs/experiments/0-baselines/`.  
These establish the performance floor/ceiling before any system innovation.

| Config file | Retrieval model | Query mode | Notes |
|-------------|-----------------|------------|-------|
| `A0_baseline_bm25_fullhist.yaml` | BM25 | Full conversation history | No rewriting |
| `A0_baseline_splade_fullhist.yaml` | SPLADE-v3 | Full history | No rewriting |
| `A1_baseline_bgem3_fullhist.yaml` | BGE-M3 | Full history | No rewriting |
| `A1_baseline_voyage_fullhist.yaml` | Voyage-3-large | Full history | No rewriting |
| `A2_baseline_bm25_rewrite.yaml` | BM25 | Oracle rewrite | GT rewrite from dataset |
| `A2_baseline_splade_rewrite.yaml` | SPLADE-v3 | Oracle rewrite | |
| `A2_baseline_bge15_rewrite.yaml` | BGE-base-1.5 | Oracle rewrite | |
| `A2_baseline_bgem3_rewrite.yaml` | BGE-M3 | Oracle rewrite | |
| `A2_baseline_voyage_rewrite.yaml` | Voyage-3-large | Oracle rewrite | Best single-model baseline |
| `replication_bm25.yaml` | BM25 | Last turn | MT-RAG paper replication |
| `replication_bge15.yaml` | BGE-base-1.5 | Last turn | |
| `replication_bgem3.yaml` | BGE-M3 | Last turn | |
| `replication_splade.yaml` | SPLADE-v3 | Last turn | |
| `replication_voyage.yaml` | Voyage-3-large | Last turn | |

**Run all baselines:**
```bash
./run.sh --category baselines -d all
# or
python scripts/run_experiment.py --category baselines --domain all
```

---

### 02-hybrid

Located in `configs/experiments/02-hybrid/`.  
Hybrid systems combine sparse (SPLADE) with dense (BGE-1.5 or Voyage-3) retrieval via **Reciprocal Rank Fusion (RRF, k=60)**.

| Config file | Sparse | Dense | Query | Notes |
|-------------|--------|-------|-------|-------|
| `hybrid_splade_voyage_norewrite.yaml` | SPLADE-v3 | Voyage-3 | Full history | No rewriting |
| `hybrid_splade_voyage_rewrite.yaml` | SPLADE-v3 | Voyage-3 | Oracle rewrite | **Best system** |
| `hybrid_splade_voyage_rewrite_v2.yaml` | SPLADE-v3 | Voyage-3 | Cohere v2 rewrite | |
| `hybrid_splade_voyage_rewrite_v3.yaml` | SPLADE-v3 | Voyage-3 | Cohere v3 rewrite | |
| `hybrid_splade_voyage_rewrite_own.yaml` | SPLADE-v3 | Voyage-3 | Own rewriter | |
| `hybrid_splade_voyage_rewrite_hyde.yaml` | SPLADE-v3 | Voyage-3 | HyDE | |
| `hybrid_splade_voyage_rewrite_multi.yaml` | SPLADE-v3 | Voyage-3 | Multi-query | |
| `hybrid_splade_bge15_norewrite.yaml` | SPLADE-v3 | BGE-base-1.5 | Full history | No rewriting |
| `hybrid_splade_bge15_rewrite.yaml` | SPLADE-v3 | BGE-base-1.5 | Oracle rewrite | |
| `hybrid_splade_bge15_rewrite_v2.yaml` | SPLADE-v3 | BGE-base-1.5 | Cohere v2 | |
| `hybrid_splade_bge15_rewrite_v3.yaml` | SPLADE-v3 | BGE-base-1.5 | Cohere v3 | |
| `hybrid_splade_bge15_rewrite_own.yaml` | SPLADE-v3 | BGE-base-1.5 | Own rewriter | |

**Fusion formula (RRF):**

$$\text{score}(d) = \sum_{r \in \{\text{sparse}, \text{dense}\}} \frac{1}{k + \text{rank}_r(d)}, \quad k = 60$$

**Run all hybrid experiments:**
```bash
./run.sh --category hybrid -d all
# or
python scripts/run_experiment.py --category hybrid --domain all
```

---

## Statistical Validation

Full statistical validation is in `statistical_results/`. All tests use:

- **Bootstrap CI** — 10 000 iterations, seed=42, metric: nDCG@5
- **Wilcoxon signed-rank** — paired, non-parametric (Shapiro-Wilk rejects normality in 100% of cases, p < 1e-8)
- **Holm-Bonferroni** — family-wise error rate correction across all comparison sets
- **Cohen's d** — effect size (negligible < 0.2, small < 0.5, medium < 0.8, large ≥ 0.8)

### Summary of results

| Hypothesis | Tests | Survive Holm-Bonferroni | Conclusion |
|------------|-------|-------------------------|------------|
| **H1: Hybrid > individual component** | 40 | **32 / 40** ✅ | Strongly supported |
| Rewrite strategy differences | 28 | 0 / 28 | Effects real but small (\|d\| < 0.21); not significant after correction |
| Turn-level degradation | 24 | 2 / 24 | Partial |
| Cross-domain rank concordance (Kendall τ) | 6 | 1 / 6 | Mean τ = 0.778 |

**Key finding:** Hybrid fusion (SPLADE + Voyage, RRF) significantly outperforms individual components in **32 of 40** paired comparisons after family-wise error correction. BM25 is significantly inferior to all neural models (all p < 1e-9).

### Regenerating statistical tests

```bash
# Full Wilcoxon + Bootstrap CI (requires per-query score files from HF)
python scripts/legacy_statistical_validation.py

# Ablation statistical tests
python scripts/run_ablation_statistical_tests.py

# All analyses
python scripts/run_all_analyses.py
```

---

## Reproducing Results

### Step-by-step (Docker, recommended)

```bash
# 1. Clone repo
git clone https://github.com/PLN-disca-iimas/mtrag_semeval2026.git
cd "mtrag_semeval2026/TASK A"

# 2. Download data from HF
python -c "
from huggingface_hub import snapshot_download
snapshot_download('vania-janet/mt-rag-benchmark-data', repo_type='dataset', local_dir='./hf_data')
"
ln -s hf_data/indices indices
ln -s hf_data/data data

# 3. Configure API keys
cp .env.example .env && nano .env

# 4. Build Docker image
docker compose build

# 5. Run ALL baselines
docker compose run --rm mtrag-retrieval \
    python scripts/run_experiment.py --category baselines --domain all

# 6. Run ALL hybrid experiments
docker compose run --rm mtrag-retrieval \
    python scripts/run_experiment.py --category hybrid --domain all

# 7. Compute metrics over all results
python scripts/run_all_analyses.py
```

### Reproducing a single result (e.g., best system)

```bash
python scripts/run_experiment.py \
    --experiment hybrid_splade_voyage_rewrite \
    --domain all
# Results written to: data/submissions/hybrid_splade_voyage_rewrite/{domain}/
```

### Expected runtime

| System | Hardware | Time (all 4 domains) |
|--------|----------|----------------------|
| BM25 baseline | CPU only | ~5 min |
| SPLADE baseline | GPU (8 GB) | ~20 min |
| BGE-1.5 baseline | GPU (8 GB) | ~15 min |
| Voyage-3 baseline | Voyage API | ~10 min (API calls) |
| Hybrid SPLADE+Voyage | GPU + API | ~25 min |

---

## API Keys Required

| Service | Variable | Required for |
|---------|----------|--------------|
| [Voyage AI](https://www.voyageai.com/) | `VOYAGE_API_KEY` | Voyage-3-large dense retrieval |
| [Cohere](https://cohere.com/) | `COHERE_API_KEY` | Cohere rewriting and reranking |
| [Hugging Face](https://huggingface.co/settings/tokens) | `HUGGINGFACE_TOKEN` | Model downloads (some gated) |

Copy `.env.example` → `.env` and fill in your keys:

```bash
cp .env.example .env
```

BM25, SPLADE, BGE-1.5, and BGE-M3 experiments **do not require any API key**.

---

## Citation

If you use this code or the benchmark artifacts, please cite:

```bibtex
@inproceedings{janet2026mtrag,
  title     = {Hybrid Sparse-Dense Retrieval for Multi-Turn Conversational Search},
  author    = {Vania Janet and {IIMAS-RAG Team}},
  booktitle = {Proceedings of SemEval 2026},
  year      = {2026},
  url       = {https://github.com/PLN-disca-iimas/mtrag_semeval2026}
}

@dataset{janet2025mtragdata,
  title  = {{MT-RAG} Benchmark Task A -- Retrieval Artifacts},
  author = {Vania Janet},
  year   = {2025},
  url    = {https://huggingface.co/datasets/vania-janet/mt-rag-benchmark-data},
  license = {CC-BY-4.0}
}
```

---

## Related Links

- 🤗 **Data & Indices:** [vania-janet/mt-rag-benchmark-data](https://huggingface.co/datasets/vania-janet/mt-rag-benchmark-data)
- 📊 **Statistical results:** [`statistical_results/`](statistical_results/)
- 🐳 **Docker image:** built from [`Dockerfile`](Dockerfile)
- 🏆 **SemEval 2026:** [Task page](https://semeval.github.io/SemEval2026/)
