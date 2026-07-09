#!/usr/bin/env python3
"""
Manual script to ingest arXiv papers into PostgreSQL and OpenSearch.

Usage:
    # From project root with venv activated:
    python scripts/manual_ingest.py
    
    # Or specify max papers:
    python scripts/manual_ingest.py --max-results 10
"""
import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.factory import make_database
from src.services.arxiv.factory import make_arxiv_client
from src.services.metadata_fetcher import make_metadata_fetcher
from src.services.opensearch.factory import make_opensearch_client
from src.services.pdf_parser.factory import make_pdf_parser_service


async def main(max_results: int = 10, process_pdfs: bool = True, index_to_opensearch: bool = True):
    """Run the ingestion pipeline."""
    print("=" * 60)
    print("Manual arXiv Paper Ingestion")
    print("=" * 60)
    
    # Initialize services
    print("\n[1/5] Initializing services...")
    arxiv_client = make_arxiv_client()
    pdf_parser = make_pdf_parser_service()
    database = make_database()
    opensearch_client = make_opensearch_client()
    metadata_fetcher = make_metadata_fetcher(arxiv_client, pdf_parser, opensearch_client)
    
    # Check OpenSearch
    print("[2/5] Checking OpenSearch...")
    if opensearch_client.health_check():
        print("  OpenSearch is healthy")
        opensearch_client.create_index(force=False)
        print("  Index ready")
    else:
        print("  WARNING: OpenSearch not available, skipping indexing")
        index_to_opensearch = False
    
    # Fetch and process papers (no date filter = get latest)
    print(f"\n[3/5] Fetching {max_results} latest CS.AI papers from arXiv...")
    
    with database.get_session() as session:
        results = await metadata_fetcher.fetch_and_process_papers(
            max_results=max_results,
            from_date=None,  # No date filter - get latest
            to_date=None,
            process_pdfs=process_pdfs,
            store_to_db=True,
            db_session=session,
            index_to_opensearch=index_to_opensearch,
        )
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Papers fetched:    {results.get('papers_fetched', 0)}")
    print(f"  PDFs downloaded:   {results.get('pdfs_downloaded', 0)}")
    print(f"  PDFs parsed:       {results.get('pdfs_parsed', 0)}")
    print(f"  Papers stored:     {results.get('papers_stored', 0)}")
    print(f"  Papers indexed:    {results.get('papers_indexed', 0)}")
    print(f"  Errors:            {len(results.get('errors', []))}")
    print(f"  Processing time:   {results.get('processing_time', 0):.1f}s")
    
    if results.get('errors'):
        print("\nErrors:")
        for err in results['errors'][:5]:
            print(f"  - {err}")
    
    print("\nDone!")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manually ingest arXiv papers")
    parser.add_argument("--max-results", type=int, default=10, help="Max papers to fetch (default: 10)")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF processing")
    parser.add_argument("--no-index", action="store_true", help="Skip OpenSearch indexing")
    
    args = parser.parse_args()
    
    asyncio.run(main(
        max_results=args.max_results,
        process_pdfs=not args.no_pdf,
        index_to_opensearch=not args.no_index,
    ))
