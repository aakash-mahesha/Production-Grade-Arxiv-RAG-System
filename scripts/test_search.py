#!/usr/bin/env python3
"""Test OpenSearch BM25 search."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.opensearch.factory import make_opensearch_client

# Initialize client
opensearch_client = make_opensearch_client()

# Check health
print("OpenSearch Health Check")
print("=" * 40)
if opensearch_client.health_check():
    print("✅ OpenSearch is healthy\n")
else:
    print("❌ OpenSearch not available")
    exit(1)

# Get stats
stats = opensearch_client.get_index_stats()
print(f"Index: {opensearch_client.index_name}")
print(f"Documents: {stats.get('document_count', 0)}")
print()

# Simple BM25 Search
print("SIMPLE BM25 SEARCH")
print("=" * 40)

search_term = "Generative AI"  # Try: "neural", "model", "transformer", "*"
print(f"Searching for: '{search_term}'\n")

results = opensearch_client.search_papers(
    query=search_term,
    size=5
)

if results.get('hits'):
    print(f"Found {results.get('total', 0)} total matches\n")
    
    for i, paper in enumerate(results['hits'], 1):
        print(f"{i}. {paper.get('title', 'Unknown')[:70]}...")
        print(f"   Score: {paper.get('score', 0):.2f}")
        print(f"   arXiv ID: {paper.get('arxiv_id', 'N/A')}")
        if paper.get('raw_text'):
            print(f"   Has content: {len(paper.get('raw_text', ''))} chars")
        print()
else:
    print("No results found.")
    if results.get('error'):
        print(f"Error: {results['error']}")
