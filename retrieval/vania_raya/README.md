# MTRAG SemEval 2026: Multi-Turn RAG Evaluation System

Implementation of a Multi-Turn Retrieval-Augmented Generation (RAG) system for the SemEval 2026 shared task on evaluating conversational RAG systems.

## Overview

This project implements a comprehensive evaluation framework for multi-turn RAG systems based on the MTRAG benchmark ([Katsis et al., 2025](https://arxiv.org/abs/2501.03468)). The system addresses key challenges in conversational AI including retrieval diversity, answerability detection, and multi-turn question understanding across four different domains.

### Key Features

- **Multi-Turn Conversation Support**: Handles complex conversational contexts with non-standalone questions and co-references
- **Multiple Retrieval Methods**: Implements dense retrieval using OpenAI embeddings with ChromaDB vector store
- **Comprehensive Evaluation**: Supports both algorithmic metrics (ROUGE-L, BERTScore) and LLM-based judges (RAGAS, RAD-Bench)
- **Multi-Domain Coverage**: Evaluates across CLAPNQ (Wikipedia), FiQA (Finance), Government, and Cloud technical documentation
- **Flexible Architecture**: Supports both local and cloud-based LLM evaluation

## Repository Structure

```
mtrag_semeval2026/
├── retrieval/
│   └── vania_raya/
│       ├── retrieval_tasks/        # Task data and qrels
│       └── scripts/
│           ├── conversations2retrieval.py    # Convert conversations to BEIR format
│           ├── create_chroma_db.py          # Build ChromaDB vector database
│           └── evaluation/
│               ├── run_retrieval_eval.py    # Evaluate retrieval performance
│               ├── run_generation_eval.py   # Evaluate generation quality
│               ├── run_algorithmic.py       # Algorithmic metrics
│               ├── judge_wrapper.py         # LLM judge wrappers
│               ├── config.yaml              # Metric configuration
│               └── requirements.txt         # Python dependencies
```

## Proposed Solution

Our approach focuses on three core components addressing the MTRAG challenge:

### 1. Retrieval System

**Vector-Based Dense Retrieval** using ChromaDB with OpenAI embeddings (`text-embedding-3-small`):
- Converts document corpora into passage-level chunks (512 tokens, 100 token overlap)
- Builds persistent vector database for efficient similarity search
- Supports query rewriting for non-standalone questions in later conversation turns

**Key Implementation**:
- `create_chroma_db.py`: Ingests JSONL corpora and creates searchable vector stores
- `simple_retrieval_test.py`: Implements retrieval with configurable top-k results
- Evaluation metrics: Recall@k and NDCG@k using `pytrec_eval`

### 2. Generation Evaluation

**Multi-Metric Assessment Framework**:
- **Algorithmic Metrics**: ROUGE-L, BERTScore (Precision/Recall), BertK-Precision for faithfulness
- **LLM Judges**: 
  - RAGAS Faithfulness and Answer Relevancy
  - RAD-Bench style reference-based evaluation
  - IDK (I Don't Know) detection for unanswerable questions
- **IDK Conditioning**: Metrics are conditioned on answerability to penalize hallucinations on unanswerable questions

### 3. Pipeline Integration

**End-to-End RAG Evaluation**:
1. **Data Preparation**: Convert MTRAG conversations to BEIR-compatible format with `conversations2retrieval.py`
2. **Retrieval**: Query ChromaDB vector store with conversation context
3. **Generation**: Provide retrieved passages to LLM for answer generation
4. **Evaluation**: Apply both reference-based and reference-less metrics

**Subtask Coverage**:
- **Subtask A (Retrieval)**: Dense retrieval with OpenAI embeddings
- **Subtask B (Generation - Reference)**: Evaluate generation quality given gold passages
- **Subtask C (Full RAG)**: End-to-end pipeline evaluation

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key (for embeddings and evaluation)
- CUDA-capable GPU (optional, for local LLM judges)

### Setup

```bash
# Clone the repository
cd mtrag_semeval2026/retrieval/vania_raya/scripts/evaluation

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_API_KEY="your-azure-key"  # If using Azure
export OPENAI_AZURE_HOST="your-azure-endpoint"
```

## Usage

### 1. Data Preparation

Convert MTRAG conversations to retrieval format:

```bash
python conversations2retrieval.py \
  -i /path/to/conversations.json \
  -o /path/to/output \
  -t -1  # -1 for last turn, 0 for full conversation
```

### 2. Build Vector Database

Create ChromaDB database from passage-level corpus:

```bash
python create_chroma_db.py \
  --input /path/to/corpus.jsonl \
  --output ./chroma_db \
  --domain clapnq \
  --embedding-model text-embedding-3-small
```

### 3. Retrieval Evaluation

Evaluate retrieval performance:

```bash
python evaluation/run_retrieval_eval.py \
  -q /path/to/queries.jsonl \
  -qrels /path/to/qrels.tsv \
  -r /path/to/results.jsonl \
  -o ./results/
```

### 4. Generation Evaluation

Run comprehensive generation evaluation:

```bash
python evaluation/run_generation_eval.py \
  -i /path/to/responses.jsonl \
  -o ./evaluation_results.jsonl \
  -e config.yaml \
  --provider openai \
  --openai_key $OPENAI_API_KEY \
  --azure_host $OPENAI_AZURE_HOST
```

For local evaluation with Hugging Face models:

```bash
python evaluation/run_generation_eval.py \
  -i /path/to/responses.jsonl \
  -o ./evaluation_results.jsonl \
  -e config.yaml \
  --provider hf \
  --judge_model meta-llama/Llama-3.1-70B-Instruct
```

## Evaluation Metrics

### Retrieval Metrics
- **Recall@k**: Fraction of relevant passages retrieved in top-k results
- **NDCG@k**: Normalized Discounted Cumulative Gain at k

### Generation Metrics
- **RBalg**: Harmonic mean of BERTScore-Recall, ROUGE-L, and BertK-Precision
- **RBllm**: LLM-based reference comparison (RAD-Bench style)
- **RLF**: Reference-less faithfulness evaluation (RAGAS)

All metrics are conditioned on answerability using an IDK judge to appropriately handle unanswerable questions.

## Expected Performance

Based on MTRAG benchmark baseline results:

| Model | Recall@5 | NDCG@5 | RBalg | RLF |
|-------|----------|---------|-------|-----|
| OpenAI Embeddings | ~0.35* | ~0.30* | - | - |
| GPT-4o (Reference) | - | - | 0.45 | 0.76 |
| GPT-4o (Full RAG) | - | - | 0.40 | 0.71 |

*Estimated based on similar dense retrievers (BGE-base: 0.30 Recall@5, 0.27 NDCG@5)

## Current Implementation Status

- [x] Data preprocessing and format conversion
- [x] ChromaDB vector database creation
- [x] Dense retrieval with OpenAI embeddings
- [ ] Algorithmic evaluation metrics
- [ ] LLM-based evaluation (RAGAS, RAD-Bench)
- [ ] IDK conditioning for answerability
- [ ] Query rewriting for non-standalone questions
- [ ] Hybrid retrieval (dense + sparse)
- [ ] Multi-hop retrieval strategies

## References

- Katsis, Y., Rosenthal, S., Fadnis, K., et al. (2025). "MTRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems." *arXiv:2501.03468*
- Kuo, Y., et al. (2024). "RAD-Bench: A Benchmark for Multi-Turn RAG Systems."
- Es, S., et al. (2024). "RAGAS: Automated Evaluation of Retrieval Augmented Generation."

## License

Apache 2.0 License - See LICENSE file for details.

