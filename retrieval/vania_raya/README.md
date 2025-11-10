# MTRAG SemEval 2026: Multi-Turn Retrieval System

Implementation of Subtask A (Retrieval) for the SemEval 2026 shared task on evaluating multi-turn conversational RAG systems.

## Overview

This project implements a dense retrieval system for multi-turn conversations based on the MTRAG benchmark ([Katsis et al., 2025](https://arxiv.org/abs/2501.03468)). The system addresses key challenges in conversational retrieval including handling non-standalone questions, maintaining conversation context, and achieving passage diversity across four different domains.

### Key Features

- **Multi-Turn Context Handling**: Processes conversational queries with co-references to previous turns
- **Dense Retrieval**: Vector-based semantic search using OpenAI embeddings with ChromaDB
- **Multi-Domain Support**: Evaluates across CLAPNQ (Wikipedia), FiQA (Finance), Government, and Cloud technical documentation
- **BEIR-Compatible Format**: Follows standard information retrieval evaluation protocols
- **Efficient Vector Storage**: Persistent ChromaDB database with configurable passage chunking

## Repository Structure

```
mtrag_semeval2026/
├── retrieval/
│   └── vania_raya/
│       ├── retrieval_tasks/              # Task data and qrels for each domain
│       └── scripts/
│           ├── conversations2retrieval.py    # Convert conversations to BEIR format
│           ├── create_chroma_db.py          # Build ChromaDB vector database
│           └── evaluation/
│               ├── run_retrieval_eval.py    # Evaluate retrieval performance
│               ├── simple_retrieval_test.py # Retrieval testing utilities
│               └── requirements.txt         # Python dependencies
```

## Proposed Solution: Dense Retrieval System

Our approach implements **vector-based dense retrieval** for Subtask A of the MTRAG challenge:

Our approach implements **vector-based dense retrieval** for Subtask A of the MTRAG challenge:

### Core Components

**1. Document Indexing**
- Processes passage-level corpora in JSONL format
- Creates passage chunks with configurable size (default: 512 tokens, 100 token overlap)
- Generates dense vector representations using OpenAI `text-embedding-3-small` model
- Stores vectors in persistent ChromaDB database for efficient similarity search

**2. Query Processing**
- Converts multi-turn conversations to BEIR-compatible query format
- Supports different conversation context strategies:
  - Last turn only (default for evaluation)
  - Full conversation history
  - Configurable turn window
- Handles question-only vs. question+answer context

**3. Semantic Search**
- Performs dense vector similarity search using cosine distance
- Retrieves top-k most relevant passages per query
- Supports batch processing for efficient evaluation

**4. Evaluation Framework**
- Computes standard IR metrics: Recall@k and NDCG@k
- Uses `pytrec_eval` for robust metric calculation
- Outputs results in BEIR format for compatibility

### Technical Implementation

**Key Scripts**:
- `create_chroma_db.py`: Builds ChromaDB vector database from passage corpora
- `conversations2retrieval.py`: Converts MTRAG conversations to queries and qrels
- `simple_retrieval_test.py`: Implements retrieval logic with OpenAI embeddings
- `run_retrieval_eval.py`: Evaluates retrieval performance using standard metrics

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key (for embeddings)

### Setup

```bash
# Navigate to the scripts directory
cd mtrag_semeval2026/retrieval/vania_raya/scripts/evaluation

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-api-key"
```

## Usage

### 1. Prepare Corpus Data

Organize your passage-level corpus in JSONL format:

```json
{"_id": "doc1_p1", "text": "Passage text here...", "title": "Document Title", "url": "http://..."}
{"_id": "doc1_p2", "text": "Another passage...", "title": "Document Title", "url": "http://..."}
```

### 2. Build Vector Database

Create ChromaDB database from passage-level corpus:

```bash
python create_chroma_db.py \
  --input /path/to/passage_corpus.jsonl \
  --output ./chroma_db \
  --domain clapnq \
  --embedding-model text-embedding-3-small \
  --max-docs 1000  # Optional: limit for testing
```

This creates a persistent vector database that can be reused for multiple queries.

### 3. Convert Conversations to Retrieval Format

Convert MTRAG conversations to BEIR-compatible format:

```bash
python conversations2retrieval.py \
  -i /path/to/conversations.json \
  -o /path/to/output_dir \
  -t -1  # -1 for last turn only, 0 for full conversation
```

This generates:
- `queries.jsonl`: Query file with conversation turns
- `qrels/dev.tsv`: Relevance judgments (query-id, corpus-id, score)

### 4. Run Retrieval

Perform retrieval using the vector database:

```bash
python simple_retrieval_test.py \
  --domain clapnq \
  --corpus_file /path/to/corpus.jsonl \
  --queries_file /path/to/queries.jsonl \
  --output_file ./retrieval_results.jsonl \
  --top_k 10
```

### 5. Evaluate Results

Evaluate retrieval performance:

```bash
python evaluation/run_retrieval_eval.py \
  -q /path/to/queries.jsonl \
  -qrels /path/to/qrels/dev.tsv \
  -r ./retrieval_results.jsonl \
  -o ./evaluation_results/
```

This outputs Recall@k and NDCG@k metrics for k ∈ {1, 3, 5, 10}.

## Evaluation Metrics

The retrieval system is evaluated using standard information retrieval metrics:

- **Recall@k**: Fraction of relevant passages retrieved in top-k results
  - Measures the system's ability to find all relevant passages
  - Computed for k ∈ {1, 3, 5, 10}

- **NDCG@k**: Normalized Discounted Cumulative Gain at position k
  - Measures ranking quality, giving higher weight to relevant passages at top positions
  - Accounts for graded relevance judgments
  - Computed for k ∈ {1, 3, 5, 10}

Both metrics are computed using `pytrec_eval` following BEIR evaluation standards.

## Expected Performance

Based on MTRAG benchmark baseline results for dense retrieval models:

| Retrieval Method | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
|-----------------|----------|-----------|---------|---------|
| BM25 (baseline) | 0.20 | 0.27 | 0.18 | 0.21 |
| BGE-base 1.5 | 0.30 | 0.38 | 0.27 | 0.30 |
| ELSER | 0.49 | 0.58 | 0.45 | 0.49 |
| OpenAI Embeddings | ~0.35* | ~0.42* | ~0.30* | ~0.33* |

*Estimated performance based on similar dense retrieval architectures

### Key Challenges

The MTRAG benchmark reveals several retrieval challenges:

- **Later Turns**: Performance drops significantly after turn 1 (0.89 → 0.47 Recall@5)
- **Non-Standalone Questions**: Questions with co-references perform 6% worse (0.48 vs 0.42 Recall@5)
- **Domain Variation**: Performance varies across domains (0.47-0.56 Recall@5)
- **Passage Diversity**: Average 16.9 unique passages per conversation requires dynamic retrieval

## Current Implementation Status

- [x] Data preprocessing and BEIR format conversion
- [x] ChromaDB vector database creation with OpenAI embeddings
- [x] Dense retrieval implementation
- [x] Standard IR metrics evaluation (Recall@k, NDCG@k)
- [x] Multi-domain support (CLAPNQ, FiQA, Govt, Cloud)
- [x] Batch processing for efficient evaluation
- [ ] Query rewriting for non-standalone questions
- [ ] Hybrid retrieval (dense + sparse BM25)
- [ ] Re-ranking strategies
- [ ] Conversation-aware context encoding

## Future Improvements

To address the challenges identified in the MTRAG benchmark:

1. **Query Rewriting**: Implement LLM-based rewriting to resolve co-references in non-standalone questions
2. **Hybrid Retrieval**: Combine dense embeddings with sparse BM25 for better first-stage retrieval
3. **Conversational Encoding**: Explore methods to better encode multi-turn context
4. **Re-ranking**: Add a second-stage re-ranker to improve precision of top-k results
5. **Domain Adaptation**: Fine-tune embeddings for specific domains (finance, technical docs)

## References

- Katsis, Y., Rosenthal, S., Fadnis, K., et al. (2025). "MTRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems." *arXiv:2501.03468* [[paper]](https://arxiv.org/abs/2501.03468) [[code]](https://github.com/ibm/mt-rag-benchmark)
- Thakur, N., et al. (2021). "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models." *NeurIPS 2021*
- Xiao, S., et al. (2023). "BGE: BAAI General Embedding." *arXiv preprint*

## License

Apache 2.0 License - See LICENSE file for details.