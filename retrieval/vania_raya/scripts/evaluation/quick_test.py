"""
Quick Test: Prueba rápida con un subset pequeño del corpus
"""

import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict
import argparse
from tqdm import tqdm

# Load environment variables
load_dotenv('/Users/vania/Documents/rag_ss/mt-rag-benchmark/src/.env')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="govt", help="Domain: clapnq, cloud, fiqa, or govt")
    parser.add_argument("--corpus_size", type=int, default=100, help="Number of documents to use from corpus")
    parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve")
    
    args = parser.parse_args()
    
    # Paths
    base_path = "/Users/vania/Documents/rag_ss/mt-rag-benchmark/src"
    corpus_file = f"{base_path}/corpora/passage_level/{args.domain}.jsonl"
    queries_file = f"{base_path}/human/retrieval_tasks/{args.domain}/{args.domain}_questions.jsonl"
    output_file = f"{base_path}/scripts/evaluation/results/quick_test_{args.domain}.jsonl"
    
    print("=" * 70)
    print("Quick Test - Retrieval con subset pequeño")
    print("=" * 70)
    print(f"Domain: {args.domain}")
    print(f"Corpus size: {args.corpus_size} documents")
    print(f"Top-K: {args.top_k}")
    print("=" * 70)
    
    # Initialize OpenAI client
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    api_key = api_key.strip("'\"")
    client = OpenAI(api_key=api_key)
    
    # Load SMALL subset of corpus
    print(f"\nLoading first {args.corpus_size} documents from corpus...")
    corpus = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= args.corpus_size:
                break
            doc = json.loads(line)
            doc_id = doc['_id']
            text = f"{doc.get('title', '')} {doc['text']}".strip()
            corpus[doc_id] = text
    
    print(f"Loaded {len(corpus)} documents")
    
    # Create embeddings for corpus
    print("\nCreating embeddings for corpus...")
    doc_ids = list(corpus.keys())
    texts = [corpus[doc_id] for doc_id in doc_ids]
    
    corpus_embeddings = {}
    batch_size = 20
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch_texts = texts[i:i+batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch_texts
        )
        batch_embeddings = [data.embedding for data in response.data]
        
        for j, embedding in enumerate(batch_embeddings):
            doc_id = doc_ids[i + j]
            corpus_embeddings[doc_id] = np.array(embedding)
    
    print(f"Created embeddings for {len(corpus_embeddings)} documents")
    
    # Load queries
    print("\nLoading queries...")
    queries = []
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            query_data = json.loads(line)
            queries.append(query_data)
    
    print(f"Loaded {len(queries)} queries")
    
    # Retrieve for each query
    print(f"\nRetrieving top-{args.top_k} documents for each query...")
    results = []
    
    for query_data in tqdm(queries, desc="Retrieving"):
        query_id = query_data['_id']
        query_text = query_data['text']
        
        # Get query embedding
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query_text
        )
        query_embedding = np.array(response.data[0].embedding)
        
        # Compute cosine similarity
        scores = {}
        for doc_id, doc_embedding in corpus_embeddings.items():
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            scores[doc_id] = float(similarity)
        
        # Get top-k
        top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:args.top_k]
        
        # Format output
        contexts = [
            {
                "document_id": doc_id,
                "score": score,
                "text": corpus[doc_id][:200]  # First 200 chars
            }
            for doc_id, score in top_docs
        ]
        
        result = {
            "task_id": query_id,
            "Collection": f"mt-rag-{args.domain}-test",
            "collection": f"mt-rag-{args.domain}-test",
            "contexts": contexts
        }
        results.append(result)
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\n✓ Results saved to: {output_file}")
    print(f"\nAhora puedes evaluar con:")
    print(f"python run_retrieval_eval.py --input_file {output_file} --output_file {output_file.replace('.jsonl', '.enriched.jsonl')}")

if __name__ == "__main__":
    main()
