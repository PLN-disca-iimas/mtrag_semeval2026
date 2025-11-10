"""
Simple Retrieval System using OpenAI Embeddings
This script creates a basic retrieval system to test metrics from run_retrieval_eval.py
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

class SimpleRetriever:
    def __init__(self, model_name="text-embedding-3-small"):
        """
        Initialize the retriever with OpenAI client
        """
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        # Remove quotes if present
        api_key = api_key.strip("'\"")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.corpus = {}
        self.corpus_embeddings = {}
        
    def load_corpus(self, corpus_file: str):
        """Load corpus from JSONL file"""
        print(f"Loading corpus from {corpus_file}...")
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                doc_id = doc['_id']
                # Combine title and text for better retrieval
                text = f"{doc.get('title', '')} {doc['text']}".strip()
                self.corpus[doc_id] = text
        print(f"Loaded {len(self.corpus)} documents")
        
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return response.data[0].embedding
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 20) -> List[List[float]]:
        """Get embeddings for multiple texts in batches"""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch
            )
            embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(embeddings)
        return all_embeddings
    
    def embed_corpus(self):
        """Create embeddings for all corpus documents"""
        print("Creating embeddings for corpus...")
        doc_ids = list(self.corpus.keys())
        texts = [self.corpus[doc_id] for doc_id in doc_ids]
        
        # Process in batches with progress bar
        batch_size = 20
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding corpus"):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.get_embeddings_batch(batch_texts, batch_size=len(batch_texts))
            all_embeddings.extend(batch_embeddings)
        
        # Store embeddings
        for doc_id, embedding in zip(doc_ids, all_embeddings):
            self.corpus_embeddings[doc_id] = np.array(embedding)
        
        print(f"Created embeddings for {len(self.corpus_embeddings)} documents")
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve top-k documents for a query"""
        # Get query embedding
        query_embedding = np.array(self.get_embedding(query))
        
        # Compute cosine similarity with all documents
        scores = {}
        for doc_id, doc_embedding in self.corpus_embeddings.items():
            # Cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            scores[doc_id] = float(similarity)
        
        # Sort by score and get top-k
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Format results
        results = [
            {"document_id": doc_id, "score": score}
            for doc_id, score in sorted_docs
        ]
        
        return results, scores

def main():
    parser = argparse.ArgumentParser(description="Simple retrieval system using OpenAI embeddings")
    parser.add_argument("--domain", type=str, default="clapnq", 
                       choices=["clapnq", "cloud", "fiqa", "govt"],
                       help="Domain to test on")
    parser.add_argument("--corpus_file", type=str, default=None,
                       help="Path to corpus file (default: auto-detect based on domain)")
    parser.add_argument("--queries_file", type=str, default=None,
                       help="Path to queries file (default: auto-detect based on domain)")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file for results (default: results/simple_retrieval_{domain}.jsonl)")
    parser.add_argument("--top_k", type=int, default=100,
                       help="Number of documents to retrieve per query")
    parser.add_argument("--collection_name", type=str, default=None,
                       help="Collection name to use in results (default: simple-{domain})")
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    base_dir = "/Users/vania/Documents/rag_ss/mt-rag-benchmark/src"
    
    if args.corpus_file is None:
        args.corpus_file = f"{base_dir}/corpora/passage_level/{args.domain}.jsonl"
    
    if args.queries_file is None:
        args.queries_file = f"{base_dir}/human/retrieval_tasks/{args.domain}/{args.domain}_questions.jsonl"
    
    if args.output_file is None:
        # Create results directory if it doesn't exist
        results_dir = f"{base_dir}/scripts/evaluation/results"
        os.makedirs(results_dir, exist_ok=True)
        args.output_file = f"{results_dir}/simple_retrieval_{args.domain}.jsonl"
    
    if args.collection_name is None:
        args.collection_name = f"simple-{args.domain}"
    
    print("=" * 80)
    print("Simple Retrieval System Test")
    print("=" * 80)
    print(f"Domain: {args.domain}")
    print(f"Corpus: {args.corpus_file}")
    print(f"Queries: {args.queries_file}")
    print(f"Output: {args.output_file}")
    print(f"Top-K: {args.top_k}")
    print("=" * 80)
    
    # Initialize retriever
    retriever = SimpleRetriever()
    
    # Load and embed corpus
    retriever.load_corpus(args.corpus_file)
    retriever.embed_corpus()
    
    # Load queries
    print(f"\nLoading queries from {args.queries_file}...")
    queries = []
    with open(args.queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            query_data = json.loads(line)
            queries.append(query_data)
    print(f"Loaded {len(queries)} queries")
    
    # Process queries and retrieve documents
    print(f"\nRetrieving top-{args.top_k} documents for each query...")
    results = []
    
    for query_data in tqdm(queries, desc="Processing queries"):
        task_id = query_data['_id']
        query_text = query_data['text']
        
        # Retrieve documents
        contexts, all_scores = retriever.retrieve(query_text, top_k=args.top_k)
        
        # Format result
        result = {
            "task_id": task_id,
            "query": query_text,
            "contexts": contexts,
            "collection": args.collection_name
        }
        results.append(result)
    
    # Save results
    print(f"\nSaving results to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\n{'=' * 80}")
    print(f"✓ Results saved to: {args.output_file}")
    print(f"✓ Total queries processed: {len(results)}")
    print(f"{'=' * 80}")
    
    print("\nTo evaluate the results, run:")
    enriched_output = args.output_file.replace('.jsonl', '.enriched.jsonl')
    print(f"\npython run_retrieval_eval.py --input_file {args.output_file} --output_file {enriched_output}")
    
if __name__ == "__main__":
    main()
