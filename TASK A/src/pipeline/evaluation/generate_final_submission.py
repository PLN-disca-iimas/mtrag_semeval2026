#!/usr/bin/env python3
"""
Generate final Task A submission for test set (rag_taskAC.jsonl).

Uses winning configurations:
- clapnq: hybrid_splade_voyage_rewrite_v2 (Cohere v2 rewrites)
- govt: hybrid_splade_voyage_rewrite_own (Cohere own rewrites)
- fiqa: hybrid_splade_voyage_rewrite_v3 (Cohere v3 rewrites)
- cloud: hybrid_splade_voyage_rewrite_v3 (Cohere v3 rewrites)

No qrels_file available, so only generates contexts without scores.
"""

import os
os.environ['HF_HOME'] = '/workspace/cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/workspace/cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/cache/transformers'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import json
import sys
from pathlib import Path
from typing import Dict, List
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.retrieval import get_sparse_retriever, get_dense_retriever, HybridRetriever
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Domain to config mapping (best performing configurations per domain)
# Using pre-generated rewrites from best models
DOMAIN_CONFIGS = {
    'clapnq': {
        'config': 'configs/experiments/02-hybrid/hybrid_splade_voyage_rewrite_v2.yaml',
        'rewrite_file': 'src/pipeline/evaluation/rewrite_final_submission/clapnq_command-r-rewrite-eval (1).jsonl',
        'description': 'Voyage + SPLADE with best rewrite model'
    },
    'govt': {
        'config': 'configs/experiments/02-hybrid/hybrid_splade_voyage_rewrite_own.yaml',
        'rewrite_file': 'src/pipeline/evaluation/rewrite_final_submission/govt_command-r-rewrite-eval (1).jsonl',
        'description': 'Voyage + SPLADE with best rewrite model'
    },
    'fiqa': {
        'config': 'configs/experiments/02-hybrid/hybrid_splade_voyage_rewrite_v3.yaml',
        'rewrite_file': 'src/pipeline/evaluation/rewrite_final_submission/fiqa_command-r-rewrite-eval (1).jsonl',
        'description': 'Voyage + SPLADE with best rewrite model'
    },
    'cloud': {
        'config': 'configs/experiments/02-hybrid/hybrid_splade_voyage_rewrite_v3.yaml',
        'rewrite_file': 'src/pipeline/evaluation/rewrite_final_submission/cloud_command-r-rewrite-eval (1).jsonl',
        'description': 'Voyage + SPLADE with best rewrite model'
    },
    # ibmcloud is aliased to cloud
    'ibmcloud': {
        'config': 'configs/experiments/02-hybrid/hybrid_splade_voyage_rewrite_v3.yaml',
        'rewrite_file': 'src/pipeline/evaluation/rewrite_final_submission/cloud_command-r-rewrite-eval (1).jsonl',
        'description': 'Voyage + SPLADE with best rewrite model (cloud alias)'
    }
}


def load_test_data(test_file: Path) -> List[Dict]:
    """Load test queries from JSONL file."""
    logger.info(f"Loading test data from {test_file}")
    queries = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line.strip()))
    logger.info(f"Loaded {len(queries)} test queries")
    return queries


def extract_query_text(input_messages: List[Dict]) -> str:
    """Extract the last user message as the query text."""
    for msg in reversed(input_messages):
        if msg.get('speaker') == 'user':
            return msg['text']
    return ""


def load_query_rewrites(rewrite_file: str) -> Dict[str, str]:
    """
    Load query rewrites from file.
    
    Args:
        rewrite_file: Path to the rewrite file
        
    Returns:
        Dict mapping task_id to rewritten query
    """
    query_map = {}
    try:
        with open(rewrite_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    query_map[item['_id']] = item['text']
        logger.info(f"Loaded {len(query_map)} rewrites from {rewrite_file}")
    except Exception as e:
        logger.warning(f"Could not load rewrites from {rewrite_file}: {e}")
    
    return query_map


def initialize_retriever(domain: str, config_path: str) -> HybridRetriever:
    """Initialize a hybrid retriever for the given domain."""
    logger.info(f"Initializing retriever for {domain} with config: {config_path}")
    
    # Map ibmcloud to cloud for file paths
    file_domain = "cloud" if domain == "ibmcloud" else domain
    
    # Load config
    config = load_config(config_path)
    
    # Initialize sparse retriever (SPLADE)
    sparse_method = config["retrieval"].get("sparse", {}).get("method", "splade")
    sparse_index = f"indices/{file_domain}/{sparse_method}"
    
    sparse_retriever = get_sparse_retriever(
        model_name=sparse_method,
        index_path=sparse_index,
        config=config["retrieval"].get("sparse", {})
    )
    logger.info(f"  ✓ Sparse retriever ({sparse_method}) initialized")
    
    # Initialize dense retriever (Voyage)
    dense_model = config["retrieval"].get("dense", {}).get("model_name", "voyage-3-large")
    
    # Handle Voyage model selection
    if "voyage" in dense_model.lower():
        if file_domain == "fiqa":
            dense_model = "voyage-finance-2"
        else:
            dense_model = "voyage-3-large"
        dense_index = f"indices/{file_domain}/voyage"
    else:
        dense_index = f"indices/{file_domain}/bge"
    
    dense_retriever = get_dense_retriever(
        model_name=dense_model,
        index_path=dense_index,
        config=config["retrieval"].get("dense", {})
    )
    logger.info(f"  ✓ Dense retriever ({dense_model}) initialized")
    
    # Create hybrid retriever
    fusion_method = config["retrieval"].get("fusion_method", "rrf")
    fusion_params = {"k": config["retrieval"].get("rrf_k", 60)}
    
    hybrid_retriever = HybridRetriever(
        sparse_retriever=sparse_retriever,
        dense_retriever=dense_retriever,
        fusion_method=fusion_method,
        fusion_params=fusion_params
    )
    logger.info(f"  ✓ Hybrid retriever initialized with {fusion_method} fusion")
    
    return hybrid_retriever, config


def generate_predictions(test_queries: List[Dict]) -> List[Dict]:
    """Generate predictions for all test queries."""
    predictions = []
    
    # Group queries by domain
    queries_by_domain = {}
    for query_item in test_queries:
        domain = query_item['Collection']
        if domain not in queries_by_domain:
            queries_by_domain[domain] = []
        queries_by_domain[domain].append(query_item)
    
    logger.info(f"\nGrouped queries into {len(queries_by_domain)} domains:")
    for domain, queries in queries_by_domain.items():
        logger.info(f"  • {domain}: {len(queries)} queries")
    
    # Process each domain
    for domain, domain_queries in queries_by_domain.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing domain: {domain.upper()}")
        logger.info(f"{'='*80}")
        
        # Get config for this domain
        domain_config = DOMAIN_CONFIGS.get(domain)
        if not domain_config:
            logger.error(f"No config found for domain: {domain}")
            for query_item in domain_queries:
                predictions.append({
                    'task_id': query_item['task_id'],
                    'Collection': domain,
                    'contexts': []
                })
            continue
        
        config_path = domain_config['config']
        rewrite_file = domain_config['rewrite_file']
        logger.info(f"Configuration: {domain_config['description']}")
        logger.info(f"Config file: {config_path}")
        logger.info(f"Rewrite file: {rewrite_file}")
        
        # Initialize retriever
        try:
            retriever, config = initialize_retriever(domain, config_path)
        except Exception as e:
            logger.error(f"Failed to initialize retriever for {domain}: {e}")
            import traceback
            traceback.print_exc()
            for query_item in domain_queries:
                predictions.append({
                    'task_id': query_item['task_id'],
                    'Collection': domain,
                    'contexts': []
                })
            continue
        
        # Load query rewrites
        query_rewrites = load_query_rewrites(rewrite_file)
        
        # Process queries for this domain
        logger.info(f"\nProcessing {len(domain_queries)} queries...")
        for i, query_item in enumerate(domain_queries, 1):
            task_id = query_item['task_id']
            input_messages = query_item['input']
            
            # Get query text
            # Try to use rewrite if available, otherwise fall back to last turn
            if query_rewrites and task_id in query_rewrites:
                query_text = query_rewrites[task_id]
                use_rewrite = True
            else:
                query_text = extract_query_text(input_messages)
                use_rewrite = False
            
            if i % 20 == 1:  # Log progress every 20 queries
                rewrite_status = "rewrite" if use_rewrite else "last_turn"
                logger.info(f"  [{i}/{len(domain_queries)}] {task_id[:40]}... ({rewrite_status})")
            
            if not query_text:
                logger.warning(f"  [{i}] No query text found for {task_id}!")
                predictions.append({
                    'task_id': task_id,
                    'Collection': domain,
                    'contexts': []
                })
                continue
            
            # Retrieve documents
            try:
                results = retriever.retrieve(query_text, top_k=10)
                
                # Format contexts - NO SCORES as requested
                contexts = []
                for result in results:
                    contexts.append({
                        'document_id': result['id'],
                        'score': float(result['score'])  # Keep score for validation
                    })
                
                predictions.append({
                    'task_id': task_id,
                    'Collection': domain,
                    'contexts': contexts
                })
                
            except Exception as e:
                logger.error(f"  [{i}] Error for {task_id}: {e}")
                predictions.append({
                    'task_id': task_id,
                    'Collection': domain,
                    'contexts': []
                })
        
        logger.info(f"✓ Completed {domain}: {len(domain_queries)} queries processed")
    
    return predictions


def save_predictions(predictions: List[Dict], output_file: Path):
    """Save predictions to JSONL file with UTF-8 encoding."""
    logger.info(f"\nSaving {len(predictions)} predictions to {output_file}")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            json.dump(pred, f, ensure_ascii=False)
            f.write('\n')
    
    logger.info(f"✓ Saved predictions to {output_file}")


def main():
    # Change to project root
    project_root = Path(__file__).parent.parent.parent.parent
    os.chdir(project_root)
    
    # Paths
    test_file = Path('src/pipeline/evaluation/rag_taskAC.jsonl')
    output_dir = Path('src/pipeline/evaluation/rewrite_final_submission')
    output_file = output_dir / 'task_a_final_submission.jsonl'
    
    logger.info("=" * 80)
    logger.info("FINAL SUBMISSION GENERATION - TASK A")
    logger.info("=" * 80)
    logger.info(f"\nWorking directory: {os.getcwd()}")
    logger.info(f"Test file: {test_file}")
    logger.info(f"Output file: {output_file}")
    
    logger.info("\nConfiguration per domain:")
    for domain, cfg in DOMAIN_CONFIGS.items():
        if domain != 'ibmcloud':  # Skip alias
            logger.info(f"  • {domain:8s}: {cfg['description']}")
    
    # Load test data
    logger.info("\n" + "=" * 80)
    test_queries = load_test_data(test_file)
    
    # Generate predictions
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING PREDICTIONS")
    logger.info("=" * 80)
    predictions = generate_predictions(test_queries)
    
    # Save predictions
    logger.info("\n" + "=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)
    save_predictions(predictions, output_file)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUBMISSION GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\n✓ Output file: {output_file}")
    logger.info(f"✓ Total predictions: {len(predictions)}")
    
    # Count by domain
    from collections import Counter
    domain_counts = Counter(p['Collection'] for p in predictions)
    logger.info(f"\nPredictions by domain:")
    for domain, count in sorted(domain_counts.items()):
        logger.info(f"  • {domain:8s}: {count:3d} queries")
    
    # Count empty predictions
    empty_count = sum(1 for p in predictions if not p['contexts'])
    if empty_count > 0:
        logger.warning(f"\n⚠ Empty predictions: {empty_count}/{len(predictions)}")
    else:
        logger.info(f"\n✓ No empty predictions!")
    
    # Validation instructions
    logger.info("\n" + "=" * 80)
    logger.info("NEXT STEPS - VALIDATION")
    logger.info("=" * 80)
    logger.info(f"\n1. Validate format:")
    logger.info(f"   python src/pipeline/evaluation/format_checker.py \\")
    logger.info(f"     --input_file {test_file} \\")
    logger.info(f"     --prediction_file {output_file} \\")
    logger.info(f"     --mode retrieval_taska")
    logger.info(f"\n2. Check sample output:")
    logger.info(f"   head -3 {output_file}")
    logger.info(f"\n3. Check file size:")
    logger.info(f"   du -h {output_file}")
    logger.info("\n" + "=" * 80)


if __name__ == '__main__':
    main()
