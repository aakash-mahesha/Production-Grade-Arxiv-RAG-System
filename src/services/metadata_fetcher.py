import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dateutil import parser as date_parser
from sqlalchemy.orm import Session
from src.config import Settings
from src.exceptions import MetadataFetchingException, PipelineException
from src.repositories.paper import PaperRepository
from src.schemas.arxiv.paper import ArxivPaper, PaperCreate
from src.schemas.pdf_parser.models import ArxivMetadata, ParsedPaper, PdfContent
from src.services.arxiv.client import ArxivClient
from src.services.opensearch.client import OpenSearchClient
from src.services.pdf_parser.parser import PDFParserService

logger = logging.getLogger(__name__)


class MetadataFetcher:
    """Service for fetching arXiv papers with PDF processing and database storage."""

    def __init__(
        self,
        arxiv_client: ArxivClient,
        pdf_parser: PDFParserService,
        opensearch_client: Optional[OpenSearchClient] = None,
        pdf_cache_dir: Optional[Path] = None,
        max_concurrent_downloads: int = 5,
        max_concurrent_parsing: int = 3,
        settings: Optional[Settings] = None,
    ):
        """Initialize metadata fetcher with services and settings."""
        from src.config import get_settings

        self.arxiv_client = arxiv_client
        self.pdf_parser = pdf_parser
        self.opensearch_client = opensearch_client
        self.pdf_cache_dir = pdf_cache_dir or self.arxiv_client.pdf_cache_dir
        self.max_concurrent_downloads = max_concurrent_downloads
        self.max_concurrent_parsing = max_concurrent_parsing
        self.settings = settings or get_settings()

    async def fetch_and_process_papers(
        self,
        max_results: Optional[int] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        process_pdfs: bool = True,
        store_to_db: bool = True,
        db_session: Optional[Session] = None,
        index_to_opensearch: bool = False,
    ) -> Dict[str, Any]:
        """
        Fetch papers from arXiv, process PDFs, and store to database/OpenSearch.
        
        SEQUENTIAL PROCESSING: Each paper is fully processed (download -> parse -> 
        store to DB -> index to OpenSearch) before moving to the next. This prevents
        memory accumulation and ensures data is saved even if process is killed.
        """
        results = {
            "papers_fetched": 0,
            "pdfs_downloaded": 0,
            "pdfs_parsed": 0,
            "papers_stored": 0,
            "papers_indexed": 0,
            "errors": [],
            "processing_time": 0,
        }

        start_time = datetime.now()

        try:
            # Step 1: Fetch paper metadata from arXiv
            logger.info("Step 1: Fetching papers from arXiv...")
            papers = await self.arxiv_client.fetch_papers(
                max_results=max_results,
                from_date=from_date,
                to_date=to_date,
                sort_by="submittedDate",
                sort_order="descending"
            )
            results["papers_fetched"] = len(papers)

            if not papers:
                logger.warning("No papers found")
                return results

            # Step 2: Process each paper SEQUENTIALLY
            # This prevents memory accumulation and ensures immediate persistence
            logger.info(f"Step 2: Processing {len(papers)} papers sequentially...")
            
            for i, paper in enumerate(papers, 1):
                logger.info(f"[{i}/{len(papers)}] Processing {paper.arxiv_id}...")
                
                try:
                    parsed_content = None
                    
                    # 2a: Download PDF
                    if process_pdfs:
                        pdf_path = await self.arxiv_client.download_pdf(paper, use_cache=True)
                        if pdf_path:
                            results["pdfs_downloaded"] += 1
                            
                            # 2b: Parse PDF
                            try:
                                pdf_content = await self.pdf_parser.parse_pdf(pdf_path)
                                if pdf_content:
                                    results["pdfs_parsed"] += 1
                                    parsed_content = pdf_content
                                    logger.info(f"  Parsed: {len(pdf_content.raw_text)} chars")
                            except Exception as parse_err:
                                # Log but don't fail - paper metadata is still valuable
                                logger.warning(f"  Parse skipped: {parse_err}")
                        else:
                            logger.warning(f"  Download failed")
                    
                    # 2c: Store to database IMMEDIATELY
                    if store_to_db and db_session:
                        try:
                            self._store_single_paper(paper, parsed_content, db_session)
                            results["papers_stored"] += 1
                            logger.info(f"  Stored to DB")
                        except Exception as db_err:
                            logger.error(f"  DB error: {db_err}")
                            results["errors"].append(f"DB error for {paper.arxiv_id}: {str(db_err)}")
                    
                    # 2d: Index to OpenSearch IMMEDIATELY
                    if index_to_opensearch and self.opensearch_client:
                        try:
                            if self._index_single_paper(paper, parsed_content):
                                results["papers_indexed"] += 1
                                logger.info(f"  Indexed to OpenSearch")
                        except Exception as os_err:
                            logger.error(f"  OpenSearch error: {os_err}")
                    
                    # Clear references to free memory
                    del parsed_content
                    
                except Exception as paper_err:
                    error_msg = f"Error processing {paper.arxiv_id}: {str(paper_err)}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
                    continue  # Continue with next paper

            # Calculate total processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            results["processing_time"] = processing_time

            logger.info(
                f"Pipeline completed in {processing_time:.1f}s: "
                f"{results['papers_fetched']} fetched, "
                f"{results['pdfs_parsed']} parsed, "
                f"{results['papers_stored']} stored, "
                f"{results['papers_indexed']} indexed, "
                f"{len(results['errors'])} errors"
            )

            return results

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            results["errors"].append(f"Pipeline error: {str(e)}")
            raise PipelineException(f"Pipeline execution failed: {e}") from e

    async def _process_pdfs_batch(self, papers: List[ArxivPaper]) -> Dict[str, Any]:
        """
        Process PDFs for a batch of papers with async concurrency.
        
        Uses overlapping download+parse pipeline for optimal throughput.
        """
        results = {
            "downloaded": 0,
            "parsed": 0,
            "parsed_papers": {},
            "errors": [],
            "download_failures": [],
            "parse_failures": [],
        }

        logger.info(f"Starting async pipeline for {len(papers)} PDFs...")

        # Create semaphores for controlled concurrency
        download_semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
        parse_semaphore = asyncio.Semaphore(self.max_concurrent_parsing)

        # Start all download+parse pipelines concurrently
        pipeline_tasks = [
            self._download_and_parse_pipeline(paper, download_semaphore, parse_semaphore)
            for paper in papers
        ]

        # Wait for all pipelines to complete
        pipeline_results = await asyncio.gather(*pipeline_tasks, return_exceptions=True)

        # Process results
        for paper, result in zip(papers, pipeline_results):
            if isinstance(result, Exception):
                error_msg = f"Pipeline error for {paper.arxiv_id}: {str(result)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
            elif result:
                download_success, parsed_paper = result

                if download_success:
                    results["downloaded"] += 1
                    if parsed_paper:
                        results["parsed"] += 1
                        results["parsed_papers"][paper.arxiv_id] = parsed_paper
                    else:
                        results["parse_failures"].append(paper.arxiv_id)
                else:
                    results["download_failures"].append(paper.arxiv_id)

        logger.info(f"PDF processing: {results['downloaded']}/{len(papers)} downloaded, {results['parsed']} parsed")
        return results

    async def _download_and_parse_pipeline(
        self,
        paper: ArxivPaper,
        download_semaphore: asyncio.Semaphore,
        parse_semaphore: asyncio.Semaphore
    ) -> tuple:
        """
        Complete download+parse pipeline for a single paper.
        
        Returns:
            Tuple of (download_success: bool, parsed_paper: Optional[ParsedPaper])
        """
        download_success = False
        parsed_paper = None

        try:
            # Step 1: Download PDF with concurrency control
            async with download_semaphore:
                logger.debug(f"Starting download: {paper.arxiv_id}")
                pdf_path = await self.arxiv_client.download_pdf(paper, False)

                if pdf_path:
                    download_success = True
                else:
                    logger.error(f"Download failed: {paper.arxiv_id}")
                    return (False, None)

            # Step 2: Parse PDF with concurrency control
            async with parse_semaphore:
                logger.debug(f"Starting parse: {paper.arxiv_id}")
                pdf_content = await self.pdf_parser.parse_pdf(pdf_path)

                if pdf_content:
                    # Create ArxivMetadata from the paper
                    arxiv_metadata = ArxivMetadata(
                        title=paper.title,
                        authors=paper.authors,
                        abstract=paper.abstract,
                        arxiv_id=paper.arxiv_id,
                        categories=paper.categories,
                        published_date=paper.published_date,
                        pdf_url=paper.pdf_url,
                    )

                    # Combine into ParsedPaper
                    parsed_paper = ParsedPaper(
                        arxiv_metadata=arxiv_metadata,
                        pdf_content=pdf_content
                    )
                    logger.debug(f"Parse complete: {paper.arxiv_id}")
                else:
                    logger.warning(f"PDF parsing failed for {paper.arxiv_id}")

        except Exception as e:
            logger.error(f"Pipeline error for {paper.arxiv_id}: {e}")
            raise MetadataFetchingException(f"Pipeline error: {e}") from e

        return (download_success, parsed_paper)

    def _serialize_parsed_content(self, parsed_paper: ParsedPaper) -> Dict[str, Any]:
        """Serialize ParsedPaper content for database storage."""
        try:
            pdf_content = parsed_paper.pdf_content

            sections = [
                {"title": section.title, "content": section.content}
                for section in pdf_content.sections
            ]
            references = list(pdf_content.references)

            return {
                "raw_text": pdf_content.raw_text,
                "sections": sections,
                "references": references,
                "parser_used": pdf_content.parser_used.value if pdf_content.parser_used else None,
                "parser_metadata": pdf_content.metadata or {},
                "pdf_processed": True,
                "pdf_processing_date": datetime.now(),
            }
        except Exception as e:
            logger.error(f"Failed to serialize parsed content: {e}")
            return {"pdf_processed": False, "parser_metadata": {"error": str(e)}}

    def _store_single_paper(
        self,
        paper: ArxivPaper,
        pdf_content: Optional[PdfContent],
        db_session: Session,
    ) -> bool:
        """Store a single paper to database immediately after processing."""
        paper_repo = PaperRepository(db_session)
        
        # Parse date
        published_date = (
            date_parser.parse(paper.published_date)
            if isinstance(paper.published_date, str)
            else paper.published_date
        )
        
        # Base paper data
        paper_data = {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "categories": paper.categories,
            "published_date": published_date,
            "pdf_url": paper.pdf_url,
        }
        
        # Add parsed content if available
        if pdf_content:
            paper_data.update({
                "raw_text": pdf_content.raw_text,
                "sections": [{"title": s.title, "content": s.content} for s in pdf_content.sections],
                "references": list(pdf_content.references),
                "parser_used": pdf_content.parser_used.value if pdf_content.parser_used else None,
                "parser_metadata": pdf_content.metadata or {},
                "pdf_processed": True,
                "pdf_processing_date": datetime.now(),
            })
        else:
            paper_data.update({
                "pdf_processed": False,
                "parser_metadata": {"note": "PDF not processed"}
            })
        
        paper_create = PaperCreate(**paper_data)
        stored = paper_repo.upsert(paper_create)
        
        # Commit immediately to persist
        db_session.commit()
        return stored is not None

    def _index_single_paper(
        self,
        paper: ArxivPaper,
        pdf_content: Optional[PdfContent],
    ) -> bool:
        """Index a single paper to OpenSearch immediately after processing."""
        if not self.opensearch_client:
            return False
        
        opensearch_data = {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": ", ".join(paper.authors) if isinstance(paper.authors, list) else paper.authors,
            "abstract": paper.abstract,
            "categories": paper.categories,
            "pdf_url": paper.pdf_url,
            "published_date": (
                paper.published_date.isoformat()
                if hasattr(paper.published_date, "isoformat")
                else str(paper.published_date)
            ),
            "raw_text": "",
        }
        
        # Add parsed content if available
        if pdf_content and pdf_content.raw_text:
            max_text_size = self.settings.opensearch.max_text_size
            opensearch_data["raw_text"] = pdf_content.raw_text[:max_text_size]
        
        return self.opensearch_client.index_paper(opensearch_data)

    def _store_papers_to_db(
        self,
        papers: List[ArxivPaper],
        parsed_papers: Dict[str, ParsedPaper],
        db_session: Session,
    ) -> int:
        """Store papers and parsed content to database."""
        paper_repo = PaperRepository(db_session)
        stored_count = 0

        for paper in papers:
            try:
                parsed_paper = parsed_papers.get(paper.arxiv_id)

                # Parse date
                published_date = (
                    date_parser.parse(paper.published_date)
                    if isinstance(paper.published_date, str)
                    else paper.published_date
                )

                # Base paper data
                paper_data = {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "abstract": paper.abstract,
                    "categories": paper.categories,
                    "published_date": published_date,
                    "pdf_url": paper.pdf_url,
                }

                # Add parsed content if available
                if parsed_paper:
                    parsed_content = self._serialize_parsed_content(parsed_paper)
                    paper_data.update(parsed_content)
                else:
                    paper_data.update({
                        "pdf_processed": False,
                        "parser_metadata": {"note": "PDF processing not available"}
                    })

                paper_create = PaperCreate(**paper_data)
                stored_paper = paper_repo.upsert(paper_create)

                if stored_paper:
                    stored_count += 1

            except Exception as e:
                logger.error(f"Failed to store paper {paper.arxiv_id}: {e}")

        # Commit all changes
        try:
            db_session.commit()
            logger.info(f"Committed {stored_count} papers to database")
        except Exception as e:
            logger.error(f"Failed to commit: {e}")
            db_session.rollback()
            stored_count = 0

        return stored_count

    def _index_papers_to_opensearch(
        self,
        papers: List[ArxivPaper],
        parsed_papers: Dict[str, ParsedPaper],
    ) -> int:
        """Index papers to OpenSearch for full-text search."""
        indexed_count = 0

        for paper in papers:
            try:
                parsed_paper = parsed_papers.get(paper.arxiv_id)

                # Prepare data for OpenSearch
                opensearch_data = {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": ", ".join(paper.authors) if isinstance(paper.authors, list) else paper.authors,
                    "abstract": paper.abstract,
                    "categories": paper.categories,
                    "pdf_url": paper.pdf_url,
                    "published_date": (
                        paper.published_date.isoformat()
                        if hasattr(paper.published_date, "isoformat")
                        else str(paper.published_date)
                    ),
                }

                # Add parsed content if available
                if parsed_paper and parsed_paper.pdf_content:
                    max_text_size = self.settings.opensearch.max_text_size
                    opensearch_data["raw_text"] = parsed_paper.pdf_content.raw_text[:max_text_size]
                else:
                    opensearch_data["raw_text"] = ""

                # Index to OpenSearch
                if self.opensearch_client.index_paper(opensearch_data):
                    indexed_count += 1
                    logger.debug(f"Indexed paper {paper.arxiv_id}")

            except Exception as e:
                logger.error(f"Error indexing paper {paper.arxiv_id}: {e}")

        logger.info(f"Indexed {indexed_count}/{len(papers)} papers to OpenSearch")
        return indexed_count


def make_metadata_fetcher(
    arxiv_client: ArxivClient,
    pdf_parser: PDFParserService,
    opensearch_client: Optional[OpenSearchClient] = None,
    pdf_cache_dir: Optional[Path] = None,
    settings: Optional[Settings] = None,
) -> MetadataFetcher:
    """Factory function to create MetadataFetcher instance."""
    from src.config import get_settings

    if settings is None:
        settings = get_settings()

    return MetadataFetcher(
        arxiv_client=arxiv_client,
        pdf_parser=pdf_parser,
        opensearch_client=opensearch_client,
        pdf_cache_dir=pdf_cache_dir,
        max_concurrent_downloads=settings.arxiv.max_concurrent_downloads,
        max_concurrent_parsing=settings.arxiv.max_concurrent_parsing,
        settings=settings,
    )