"""
Script to create a ChromaDB vector database from passage-level JSONL data.
This prepares your data for the RAG system with reranking.
"""

import os
import json
import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
from typing import List
import argparse

# Load environment variables
load_dotenv()

def load_passages_from_jsonl(jsonl_path: str, max_docs: int = None) -> List[Document]:
    """
    Load passages from a JSONL file and convert them to LangChain Documents.
    
    Args:
        jsonl_path: Path to the JSONL file containing passages
        max_docs: Maximum number of documents to load (for testing). None = load all.
    
    Returns:
        List of LangChain Document objects
    """
    documents = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_docs and i >= max_docs:
                break
                
            item = json.loads(line)
            
            # Extract text content (this is the passage/chunk)
            text = item.get('text', '')
            
            # Extract metadata
            metadata = {
                'id': item.get('id', item.get('_id', '')),
                'title': item.get('title', ''),
                'url': item.get('url', ''),
                'source': os.path.basename(jsonl_path)
            }
            
            # Create LangChain Document
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
    
    return documents


def create_chroma_db(
    jsonl_path: str,
    persist_directory: str,
    collection_name: str = "passages",
    max_docs: int = None,
    embedding_model: str = "text-embedding-3-small"
):
    """
    Create a ChromaDB vector store from JSONL passage data.
    
    Args:
        jsonl_path: Path to JSONL file with passage data
        persist_directory: Directory where ChromaDB will persist data
        collection_name: Name for the ChromaDB collection
        max_docs: Maximum documents to process (None = all)
        embedding_model: OpenAI embedding model to use
    """
    
    print(f"Loading passages from {jsonl_path}...")
    documents = load_passages_from_jsonl(jsonl_path, max_docs=max_docs)
    print(f"Loaded {len(documents)} documents")
    
    # Initialize OpenAI embeddings
    print(f"Initializing OpenAI embeddings with model: {embedding_model}...")
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )
    
    # Create ChromaDB vector store
    print(f"Creating ChromaDB vector store in {persist_directory}...")
    print(f"Collection name: {collection_name}")
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    print(f"✓ Successfully created ChromaDB with {len(documents)} documents")
    print(f"✓ Vector store persisted to: {persist_directory}")
    
    return vectorstore


def load_existing_chroma_db(
    persist_directory: str,
    collection_name: str = "passages",
    embedding_model: str = "text-embedding-3-small"
):
    """
    Load an existing ChromaDB vector store.
    
    Args:
        persist_directory: Directory where ChromaDB is persisted
        collection_name: Name of the ChromaDB collection
        embedding_model: OpenAI embedding model (must match the one used to create)
    """
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    
    print(f"✓ Loaded existing ChromaDB from {persist_directory}")
    return vectorstore


def main():
    parser = argparse.ArgumentParser(
        description="Create or load a ChromaDB vector database from passage-level JSONL data"
    )
    parser.add_argument(
        "--jsonl_path",
        type=str,
        required=False,
        help="Path to JSONL file with passage data (required for creation)"
    )
    parser.add_argument(
        "--persist_directory",
        type=str,
        default="./chroma_db",
        help="Directory where ChromaDB will persist data (default: ./chroma_db)"
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="passages",
        help="Name for the ChromaDB collection (default: passages)"
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="Maximum number of documents to process (for testing). Default: all documents"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model to use (default: text-embedding-3-small)"
    )
    parser.add_argument(
        "--load_only",
        action="store_true",
        help="Only load existing database, don't create new one"
    )
    
    args = parser.parse_args()
    
    # Verify OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError(
            "OPENAI_API_KEY not found in environment. "
            "Make sure your .env file contains OPENAI_API_KEY=your_key_here"
        )
    
    if args.load_only:
        # Load existing database
        vectorstore = load_existing_chroma_db(
            persist_directory=args.persist_directory,
            collection_name=args.collection_name,
            embedding_model=args.embedding_model
        )
    else:
        # Create new database
        if not args.jsonl_path:
            raise ValueError("--jsonl_path is required when creating a new database")
        
        if not os.path.exists(args.jsonl_path):
            raise FileNotFoundError(f"JSONL file not found: {args.jsonl_path}")
        
        vectorstore = create_chroma_db(
            jsonl_path=args.jsonl_path,
            persist_directory=args.persist_directory,
            collection_name=args.collection_name,
            max_docs=args.max_docs,
            embedding_model=args.embedding_model
        )
    
    # Test the vector store with a sample query
    print("\n" + "="*50)
    print("Testing vector store with sample query...")
    test_query = "What is the French Revolution?"
    results = vectorstore.similarity_search(test_query, k=3)
    
    print(f"\nQuery: {test_query}")
    print(f"Retrieved {len(results)} documents:")
    for i, doc in enumerate(results, 1):
        print(f"\n--- Document {i} ---")
        print(f"Title: {doc.metadata.get('title', 'N/A')}")
        print(f"Text preview: {doc.page_content[:200]}...")
    
    return vectorstore


if __name__ == "__main__":
    main()

