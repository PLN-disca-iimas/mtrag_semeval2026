# TASK C — End-to-End RAG Pipeline

Full pipeline that integrates **Task A retrieval** (hybrid SPLADE + Voyage) with **Task B generation** (Cohere Command-R) to produce grounded, multi-turn conversational responses.

## Contents

| File | Description |
|------|-------------|
| `RAG_Completo.ipynb` | Complete RAG pipeline notebook |
| `IIMAS-RAG_taskC.jsonl` | Official task submission |

## Usage

Open `RAG_Completo.ipynb` and run all cells. Requires:
- Valid `COHERE_API_KEY` and `VOYAGE_API_KEY` in environment
- Task A retrieval results (see [`../TASK A/`](../TASK%20A/))

## Data

Retrieval artifacts (indices, corpora) available at:  
🤗 [`vania-janet/mt-rag-benchmark-data`](https://huggingface.co/datasets/vania-janet/mt-rag-benchmark-data)
