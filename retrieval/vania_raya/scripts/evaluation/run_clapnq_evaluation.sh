#!/bin/bash

# Script para ejecutar evaluación completa de ClapNQ
# Uso: ./run_clapnq_evaluation.sh [reranker|cross_encoder]

set -e  # Salir si hay errores

# Configuración
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL_DIR="$SCRIPT_DIR"
CLAPNQ_DIR="$PROJECT_ROOT/human/retrieval_tasks/clapnq"
CHROMA_DB="$PROJECT_ROOT/scripts/chroma_db"
COLLECTION_NAME="passages"
RETRIEVER_TYPE="${1:-reranker}"  # reranker o cross_encoder
TOP_K=10

# Colores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== ClapNQ Evaluation Pipeline ===${NC}\n"

# Paso 1: Convertir queries
echo -e "${YELLOW}[1/3] Converting queries...${NC}"
QUERIES_FILE="$EVAL_DIR/queries_clapnq.jsonl"
python "$EVAL_DIR/convert_queries_for_reranking.py" \
  --input_file "$CLAPNQ_DIR/clapnq_rewrite.jsonl" \
  --output_file "$QUERIES_FILE" \
  --format clapnq_rewrite

if [ ! -f "$QUERIES_FILE" ]; then
    echo "Error: Failed to create queries file"
    exit 1
fi

QUERY_COUNT=$(wc -l < "$QUERIES_FILE")
echo -e "${GREEN}✓ Converted $QUERY_COUNT queries${NC}\n"

# Paso 2: Ejecutar reranking
echo -e "${YELLOW}[2/3] Running reranking with $RETRIEVER_TYPE...${NC}"
RESULTS_FILE="$EVAL_DIR/reranking_results_clapnq.jsonl"
python "$EVAL_DIR/tests/reranking/RAG_TECHNIQUES/all_rag_techniques_runnable_scripts/reranking.py" \
  --queries_file "$QUERIES_FILE" \
  --output_file "$RESULTS_FILE" \
  --chroma_db_path "$CHROMA_DB" \
  --collection_name "$COLLECTION_NAME" \
  --retriever_type "$RETRIEVER_TYPE" \
  --top_k "$TOP_K" \
  --collection "mt-rag-clapnq-elser-512-100-20240503"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "Error: Failed to create results file"
    exit 1
fi

echo -e "${GREEN}✓ Reranking completed${NC}\n"

# Paso 3: Evaluar resultados
echo -e "${YELLOW}[3/3] Evaluating results...${NC}"
EVALUATED_FILE="$EVAL_DIR/evaluated_results_clapnq.jsonl"
python "$EVAL_DIR/run_retrieval_eval.py" \
  --input_file "$RESULTS_FILE" \
  --output_file "$EVALUATED_FILE"

if [ ! -f "$EVALUATED_FILE" ]; then
    echo "Error: Failed to create evaluated results file"
    exit 1
fi

# Mostrar resultados agregados
AGGREGATE_FILE="${EVALUATED_FILE%.jsonl}_aggregate.csv"
if [ -f "$AGGREGATE_FILE" ]; then
    echo -e "\n${GREEN}=== Evaluation Results ===${NC}"
    cat "$AGGREGATE_FILE"
    echo ""
fi

echo -e "${GREEN}✓ Evaluation completed!${NC}"
echo -e "\n${BLUE}Output files:${NC}"
echo -e "  Queries: $QUERIES_FILE"
echo -e "  Results: $RESULTS_FILE"
echo -e "  Evaluated: $EVALUATED_FILE"
if [ -f "$AGGREGATE_FILE" ]; then
    echo -e "  Aggregate: $AGGREGATE_FILE"
fi

