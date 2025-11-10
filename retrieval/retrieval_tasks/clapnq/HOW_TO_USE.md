# Cómo Usar los Archivos de Evaluación ClapNQ

Este directorio contiene los archivos necesarios para evaluar tu sistema RAG con el dataset ClapNQ.

## Archivos Disponibles

1. **`clapnq_rewrite.jsonl`**: Queries rewriteadas para evaluación (208 queries)
2. **`qrels/dev.tsv`**: Ground truth - relevancia de documentos para cada query
3. **`clapnq_questions.jsonl`**: (vacío) Preguntas originales
4. **`clapnq_lastturn.jsonl`**: (vacío) Último turno de conversaciones

## Flujo de Trabajo Completo

### Paso 1: Convertir Queries al Formato Correcto

Los archivos de evaluación usan un formato diferente. Convierte `clapnq_rewrite.jsonl` al formato que espera `reranking.py`:

```bash
cd /Users/vania/Documents/rag_ss/mt-rag-benchmark/src/scripts/evaluation

python convert_queries_for_reranking.py \
  --input_file /Users/vania/Documents/rag_ss/mt-rag-benchmark/src/human/retrieval_tasks/clapnq/clapnq_rewrite.jsonl \
  --output_file queries_for_reranking.jsonl \
  --format clapnq_rewrite
```

Esto creará `queries_for_reranking.jsonl` con el formato:
```json
{"task_id": "dd6b6ffd177f2b311abe676261279d2f::2", "query": "Where do the Arizona Cardinals play, regardless of location, this week?"}
```

### Paso 2: Ejecutar Reranking

Ejecuta el script de reranking con las queries convertidas:

```bash
python reranking.py \
  --queries_file queries_for_reranking.jsonl \
  --output_file reranking_results.jsonl \
  --chroma_db_path /Users/vania/Documents/rag_ss/mt-rag-benchmark/src/scripts/chroma_db \
  --collection_name passages \
  --retriever_type reranker \
  --top_k 10 \
  --collection mt-rag-clapnq-elser-512-100-20240503
```

**Parámetros importantes:**
- `--chroma_db_path`: Ruta a tu ChromaDB
- `--collection_name`: Nombre de la colección en ChromaDB (default: "passages")
- `--retriever_type`: `reranker` (LLM-based) o `cross_encoder` (más rápido)
- `--top_k`: Número de documentos a recuperar (10 es estándar)
- `--collection`: Debe coincidir con el nombre usado en `run_retrieval_eval.py`

### Paso 3: Evaluar Resultados

El script `run_retrieval_eval.py` automáticamente detecta el dominio "clapnq" y usa el archivo `qrels/dev.tsv` correspondiente:

```bash
python run_retrieval_eval.py \
  --input_file reranking_results.jsonl \
  --output_file evaluated_results.jsonl
```

El script:
1. Detecta automáticamente que es "clapnq" por el nombre de la colección
2. Carga `qrels/dev.tsv` desde este directorio
3. Calcula métricas: NDCG@5, NDCG@10, Recall@5, Recall@10
4. Genera un CSV con resultados agregados: `evaluated_results_aggregate.csv`

## Estructura de Archivos

### clapnq_rewrite.jsonl
```json
{"_id": "task_id::turn", "text": "query text"}
```

### qrels/dev.tsv
```
query-id	corpus-id	score
dd6b6ffd177f2b311abe676261279d2f::2	822086267_7384-8758-0-1374	1
```
- `query-id`: ID de la query (debe coincidir con `task_id` en resultados)
- `corpus-id`: ID del documento relevante
- `score`: Relevancia (1 = relevante, 0 = no relevante)

### Formato de Salida de reranking.py
```json
{
  "task_id": "dd6b6ffd177f2b311abe676261279d2f::2",
  "collection": "mt-rag-clapnq-elser-512-100-20240503",
  "contexts": [
    {
      "document_id": "822086267_7384-8758-0-1374",
      "text": "...",
      "title": "...",
      "url": "...",
      "score": 8.5
    }
  ],
  "input": [{"speaker": "user", "text": "query text"}]
}
```

## Ejemplo Completo (One-Liner)

```bash
# 1. Convertir queries
python convert_queries_for_reranking.py \
  --input_file ../../human/retrieval_tasks/clapnq/clapnq_rewrite.jsonl \
  --output_file queries.jsonl \
  --format clapnq_rewrite

# 2. Ejecutar reranking
python reranking.py \
  --queries_file queries.jsonl \
  --output_file results.jsonl \
  --chroma_db_path ../../scripts/chroma_db \
  --collection_name passages \
  --retriever_type reranker

# 3. Evaluar
python run_retrieval_eval.py \
  --input_file results.jsonl \
  --output_file evaluated_results.jsonl
```

## Notas Importantes

1. **IDs deben coincidir**: El `task_id` en los resultados debe coincidir exactamente con el `query-id` en `qrels/dev.tsv`

2. **Collection name**: El nombre de la colección en el output debe contener "clapnq" para que `run_retrieval_eval.py` detecte correctamente el dominio

3. **Document IDs**: Los `document_id` en los resultados deben coincidir con los `corpus-id` en los qrels

4. **Scores**: Los scores de reranking son de relevancia (no necesariamente 0-1), pero se usan para ordenar

## Troubleshooting

**Problema**: `run_retrieval_eval.py` no encuentra el archivo qrels
- **Solución**: Verifica que el nombre de la colección contenga "clapnq" y que el archivo `qrels/dev.tsv` exista

**Problema**: IDs no coinciden
- **Solución**: Verifica que el formato de `task_id` sea exactamente el mismo (incluyendo `::` para multi-turn)

**Problema**: No hay resultados en la evaluación
- **Solución**: Verifica que los `document_id` en los resultados coincidan con los `corpus-id` en los qrels

