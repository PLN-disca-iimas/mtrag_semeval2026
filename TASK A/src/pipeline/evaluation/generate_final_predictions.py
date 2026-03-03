#!/usr/bin/env python3
"""
Generate final predictions for Task A submission using hybrid voyage+splade retrieval
with domain-specific query rewrites.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Set cache directories
os.environ['HF_HOME'] = '/workspace/cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/workspace/cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/cache/transformers'
os.environ['HF_HUB_CACHE'] = '/workspace/cache/huggingface'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import json
import logging
from typing import Dict, List, Any
from tqdm import tqdm

from src.pipeline.retrieval import get_sparse_retriever, get_dense_retriever
from src.pipeline.retrieval.fusion import reciprocal_rank_fusion

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_queries(query_file: Path) -> Dict[str, Dict[str, Any]]:
    """Load queries from rag_taskAC.jsonl file."""
    queries = {}
    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            task_id = data['task_id']
            collection = data['Collection']
            
            # Get last user query from conversation
            last_user_query = None
            for msg in reversed(data['input']):
                if msg['speaker'] == 'user':
                    last_user_query = msg['text']
                    break
            
            queries[task_id] = {
                'original_query': last_user_query,
                'collection': collection,
                'conversation_history': data['input']
            }
    
    logger.info(f"Loaded {len(queries)} queries from {query_file}")
    return queries


def load_rewrites(rewrite_dir: Path) -> Dict[str, str]:
    """Load query rewrites for each domain."""
    rewrite_files = {
        'clapnq': 'clapnq_command-r-rewrite-evalAC.txt',
        'fiqa': 'fiqa_command-r-rewrite-evalAC.txt',
        'govt': 'govt_command-r-rewrite-evalAC.txt',
        'cloud': 'cloud_command-r-rewrite-evalAC.txt'
    }
    
    rewrites = {}
    for domain, filename in rewrite_files.items():
        filepath = rewrite_dir / filename
        if not filepath.exists():
            logger.warning(f"Rewrite file not found: {filepath}")
            continue
        
        domain_count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                rewrites[data['_id']] = data['text']
                domain_count += 1
        
        logger.info(f"Loaded {domain_count} rewrites for {domain}")
    
    return rewrites


def initialize_retrievers(domain: str):
    """Initialize SPLADE and Voyage retrievers for a domain."""
    
    # Map domain names (ibmcloud in queries -> cloud in indices)
    domain_map = {
        'ibmcloud': 'cloud',
        'clapnq': 'clapnq',
        'fiqa': 'fiqa',
        'govt': 'govt'
    }
    index_domain = domain_map.get(domain, domain)
    
    # Determine Voyage model based on domain
    if index_domain == 'fiqa':
        voyage_model = 'voyage-finance-2'
        voyage_index = 'voyage'
    else:
        voyage_model = 'voyage-3-large'
        voyage_index = 'voyage'
    
    sparse_index_path = project_root / f"indices/{index_domain}/splade"
    dense_index_path = project_root / f"indices/{index_domain}/{voyage_index}"
    
    logger.info(f"Initializing SPLADE retriever for {domain} (using {index_domain} indices)...")
    logger.info(f"  Index path: {sparse_index_path}")
    sparse_retriever = get_sparse_retriever(
        model_name='splade',
        index_path=sparse_index_path,
        config={}
    )
    
    logger.info(f"Initializing {voyage_model} retriever for {domain} (using {index_domain} indices)...")
    logger.info(f"  Index path: {dense_index_path}")
    dense_retriever = get_dense_retriever(
        model_name=voyage_model,
        index_path=dense_index_path,
        config={'model_name': voyage_model}
    )
    
    return sparse_retriever, dense_retriever


def retrieve_with_hybrid(
    query: str,
    sparse_retriever,
    dense_retriever,
    top_k: int = 300,
    rrf_k: int = 60,
    final_k: int = 10
) -> List[dict]:
    """Perform hybrid retrieval with RRF fusion."""
    
    # Get sparse results
    sparse_results = sparse_retriever.retrieve(query, top_k=top_k)
    
    # Get dense results  
    dense_results = dense_retriever.retrieve(query, top_k=top_k)
    
    # Apply RRF fusion
    fused_results = reciprocal_rank_fusion(
        [sparse_results, dense_results],
        k=rrf_k
    )
    
    # Return top final_k with document_id and score
    return [
        {'document_id': result['id'], 'score': result['score']} 
        for result in fused_results[:final_k]
    ]


def main():
    """Main execution function."""
    
    # Define paths
    query_file = project_root / "src/pipeline/evaluation/rag_taskAC.jsonl"
    rewrite_dir = project_root / "src/pipeline/evaluation/rewrite_final_submission"
    output_dir = project_root / "experiments/final_submission_taskA"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "submission_hybrid_voyage_splade.jsonl"
    
    logger.info("="*80)
    logger.info("Generating Final Predictions for Task A")
    logger.info("="*80)
    
    # Load queries and rewrites
    logger.info("Loading queries and rewrites...")
    queries = load_queries(query_file)
    rewrites = load_rewrites(rewrite_dir)
    
    # Group queries by domain
    queries_by_domain = {}
    for task_id, query_data in queries.items():
        domain = query_data['collection']
        if domain not in queries_by_domain:
            queries_by_domain[domain] = {}
        queries_by_domain[domain][task_id] = query_data
    
    logger.info(f"Queries by domain: {[(d, len(q)) for d, q in queries_by_domain.items()]}")
    
    # Process each domain
    all_predictions = []
    
    for domain in sorted(queries_by_domain.keys()):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing domain: {domain.upper()}")
        logger.info(f"{'='*80}")
        
        domain_queries = queries_by_domain[domain]
        logger.info(f"Total queries: {len(domain_queries)}")
        
        # Initialize retrievers for this domain
        try:
            sparse_retriever, dense_retriever = initialize_retrievers(domain)
        except Exception as e:
            logger.error(f"Failed to initialize retrievers for {domain}: {e}")
            logger.exception("Full traceback:")
            continue
        
        # Process each query in this domain
        for task_id, query_data in tqdm(domain_queries.items(), desc=f"Processing {domain}"):
            # Use rewritten query if available, otherwise use original
            query_text = rewrites.get(task_id, query_data['original_query'])
            
            try:
                # Retrieve top 10 documents
                contexts = retrieve_with_hybrid(
                    query=query_text,
                    sparse_retriever=sparse_retriever,
                    dense_retriever=dense_retriever,
                    top_k=300,
                    rrf_k=60,
                    final_k=10
                )
                
                # Create prediction entry
                prediction = {
                    'task_id': task_id,
                    'Collection': domain,
                    'contexts': contexts
                }
                all_predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Error processing {task_id}: {e}")
                # Add empty prediction to maintain count
                all_predictions.append({
                    'task_id': task_id,
                    'Collection': domain,
                    'contexts': []
                })
                
                # Show first few errors with traceback
                if len([p for p in all_predictions if p['contexts'] == []]) <= 3:
                    import traceback
                    traceback.print_exc()
    
    # Write predictions to file
    logger.info(f"\n{'='*80}")
    logger.info(f"Writing predictions to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred in all_predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    
    logger.info(f"Total predictions: {len(all_predictions)}")
    logger.info(f"Output saved to: {output_file}")
    
    # Run format checker
    logger.info(f"\n{'='*80}")
    logger.info("Running format checker...")
    logger.info(f"{'='*80}")
    
    format_checker_path = project_root / "src/pipeline/evaluation/format_checker.py"
    input_file = query_file
    os.system(f"python {format_checker_path} --input_file {input_file} --prediction_file {output_file} --mode retrieval_taska")
    
    logger.info("\n✅ Done! Submission file ready at:")
    logger.info(f"   {output_file}")


if __name__ == "__main__":
    main()
