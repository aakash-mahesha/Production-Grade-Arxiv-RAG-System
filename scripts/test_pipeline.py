import asyncio
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timezone
from dateutil.parser import parse as parse_date

from src.database import get_db_session
from src.services.arxiv.factory import make_arxiv_client
from src.services.pdf_parser.factory import make_pdf_parser_service
from src.repositories.paper import PaperRepository
from src.schemas.arxiv.paper import PaperCreate


async def full_pipeline_test():
    print("=" * 60)
    print("🚀 FULL PIPELINE TEST: ArXiv -> PDF Parse -> Database")
    print("=" * 60)

    # Initialize services
    arxiv_client = make_arxiv_client()
    pdf_parser = make_pdf_parser_service()

    # Step 1: Fetch papers from arXiv
    print("\n📥 Step 1: Fetching papers from arXiv...")
    papers = await arxiv_client.fetch_papers(max_results=2)
    print(f"   Found {len(papers)} papers")

    for paper in papers:
        print(f"\n" + "=" * 50)
        print(f"📄 Processing: {paper.arxiv_id}")
        print(f"   Title: {paper.title[:50]}...")

        # Step 2: Download PDF
        print(f"\n   📥 Step 2: Downloading PDF...")
        pdf_path = await arxiv_client.download_pdf(paper)

        if not pdf_path:
            print(f"   ❌ Failed to download PDF")
            continue

        print(f"   ✅ Downloaded: {pdf_path.name}")

        # Step 3: Parse PDF
        print(f"\n   📄 Step 3: Parsing PDF with Docling...")
        try:
            pdf_content = await pdf_parser.parse_pdf(pdf_path)

            if pdf_content:
                print(f"   ✅ Parsed successfully!")
                print(f"      - Sections: {len(pdf_content.sections)}")
                print(f"      - Raw text: {len(pdf_content.raw_text)} chars")
                print(f"      - Parser: {pdf_content.parser_used}")
            else:
                print(f"   ⚠️ PDF skipped (size/page limits)")
                pdf_content = None

        except Exception as e:
            print(f"   ❌ Parse error: {e}")
            pdf_content = None

        # Step 4: Save to database
        print(f"\n   💾 Step 4: Saving to database...")

        # Build PaperCreate schema
        paper_data = PaperCreate(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            authors=paper.authors,
            abstract=paper.abstract,
            categories=paper.categories,
            published_date=parse_date(paper.published_date),
            pdf_url=paper.pdf_url,
            # PDF content (if parsed)
            raw_text=pdf_content.raw_text if pdf_content else None,
            sections=[s.model_dump() for s in pdf_content.sections] if pdf_content else None,
            parser_used=pdf_content.parser_used.value if pdf_content else None,
            parser_metadata=pdf_content.metadata if pdf_content else None,
            pdf_processed=pdf_content is not None,
            pdf_processing_date=datetime.now(timezone.utc) if pdf_content else None,
        )

        with get_db_session() as session:
            repo = PaperRepository(session)
            saved_paper = repo.upsert(paper_data)
            print(f"   ✅ Saved with ID: {saved_paper.id}")
            print(f"      - pdf_processed: {saved_paper.pdf_processed}")

    # Step 5: Show stats
    print(f"\n" + "=" * 60)
    print("📊 FINAL STATS")
    print("=" * 60)

    with get_db_session() as session:
        repo = PaperRepository(session)
        stats = repo.get_processing_stats()

        print(f"   Total papers: {stats['total_papers']}")
        print(f"   Processed: {stats['processed_papers']}")
        print(f"   With text: {stats['papers_with_text']}")
        print(f"   Processing rate: {stats['processing_rate']:.1f}%")

    print("\n✅ Pipeline test complete!")


asyncio.run(full_pipeline_test())
