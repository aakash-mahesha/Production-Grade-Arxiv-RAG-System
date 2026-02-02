import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError, RequestError
from src.config import Settings, get_settings

from .index_config import ARXIV_PAPERS_INDEX, ARXIV_PAPERS_MAPPING
from .query_builder import PaperQueryBuilder

logger = logging.getLogger(__name__)


class OpenSearchClient:
    """
    Client for OpenSearch operations including index management and search.
    """

    def __init__(self, host: str = "http://localhost:9200", settings: Optional[Settings] = None):
        """Initialize OpenSearch client."""
        self.host = host
        self.settings = settings or get_settings()
        
        # Create the low-level client
        self.client = OpenSearch(
            hosts=[host],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
        
        # Use configured index name
        self.index_name = self.settings.opensearch.index_name or ARXIV_PAPERS_INDEX
        logger.info(f"OpenSearch client initialized with host: {host}")

    # ============================================================
    # INDEX MANAGEMENT
    # ============================================================

    def create_index(self, force: bool = False) -> bool:
        """
        Create the arxiv-papers index with proper mappings.

        Args:
            force: If True, delete existing index before creating

        Returns:
            True if index was created, False if it already exists
        """
        try:
            # Check if index exists
            if self.client.indices.exists(index=self.index_name):
                if force:
                    logger.info(f"Deleting existing index: {self.index_name}")
                    self.client.indices.delete(index=self.index_name)
                else:
                    logger.info(f"Index {self.index_name} already exists")
                    return False

            # Create index with mappings
            response = self.client.indices.create(
                index=self.index_name,
                body=ARXIV_PAPERS_MAPPING
            )

            if response.get("acknowledged"):
                logger.info(f"Successfully created index: {self.index_name}")
                return True
            else:
                logger.error(f"Failed to create index: {response}")
                return False

        except RequestError as e:
            logger.error(f"Error creating index: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error creating index: {e}")
            return False

    # ============================================================
    # DOCUMENT INDEXING
    # ============================================================

    def index_paper(self, paper_data: Dict[str, Any]) -> bool:
        """
        Index a single paper document.

        Args:
            paper_data: Paper data to index

        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate required field
            if "arxiv_id" not in paper_data:
                logger.error("Missing arxiv_id in paper data")
                return False

            # Add timestamps if not present
            if "created_at" not in paper_data:
                paper_data["created_at"] = datetime.now(timezone.utc).isoformat()
            if "updated_at" not in paper_data:
                paper_data["updated_at"] = datetime.now(timezone.utc).isoformat()

            # Convert authors list to string if needed
            if isinstance(paper_data.get("authors"), list):
                paper_data["authors"] = ", ".join(paper_data["authors"])

            # Index the document (use arxiv_id as document ID)
            response = self.client.index(
                index=self.index_name,
                id=paper_data["arxiv_id"],
                body=paper_data,
                refresh=True,  # Make immediately searchable
            )

            if response.get("result") in ["created", "updated"]:
                logger.debug(f"Indexed paper: {paper_data['arxiv_id']}")
                return True
            else:
                logger.error(f"Failed to index paper: {response}")
                return False

        except Exception as e:
            logger.error(f"Error indexing paper {paper_data.get('arxiv_id', 'unknown')}: {e}")
            return False

    def bulk_index_papers(self, papers: List[Dict[str, Any]]) -> Dict[str, int]:
        """Bulk index multiple papers."""
        results = {"success": 0, "failed": 0}

        for paper in papers:
            if self.index_paper(paper):
                results["success"] += 1
            else:
                results["failed"] += 1

        logger.info(f"Bulk indexing: {results['success']} success, {results['failed']} failed")
        return results

    # ============================================================
    # SEARCH
    # ============================================================

    def search_papers(
        self,
        query: str,
        size: int = 10,
        from_: int = 0,
        fields: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        track_total_hits: bool = True,
        latest_papers: bool = False,
    ) -> Dict[str, Any]:
        """
        Search papers using BM25 scoring.

        Args:
            query: Search query text
            size: Number of results to return
            from_: Offset for pagination
            fields: List of fields to search in
            categories: Filter by categories
            track_total_hits: Whether to track total hits
            latest_papers: Sort by date instead of relevance

        Returns:
            Search results with hits and metadata
        """
        try:
            # Build query using query builder
            query_builder = PaperQueryBuilder(
                query=query,
                size=size,
                from_=from_,
                fields=fields,
                categories=categories,
                track_total_hits=track_total_hits,
                latest_papers=latest_papers,
            )

            search_body = query_builder.build()

            # Execute search
            response = self.client.search(
                index=self.index_name,
                body=search_body
            )

            # Format results
            results = {
                "total": response["hits"]["total"]["value"],
                "hits": []
            }

            for hit in response["hits"]["hits"]:
                paper = hit["_source"]
                paper["score"] = hit["_score"]
                if "highlight" in hit:
                    paper["highlights"] = hit["highlight"]
                results["hits"].append(paper)

            logger.info(f"Search for '{query}' returned {results['total']} results")
            return results

        except NotFoundError:
            logger.error(f"Index {self.index_name} not found")
            return {"total": 0, "hits": [], "error": "Index not found"}
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {"total": 0, "hits": [], "error": str(e)}

    # ============================================================
    # HEALTH & STATS
    # ============================================================

    def health_check(self) -> bool:
        """Check if OpenSearch is healthy and accessible."""
        try:
            health = self.client.cluster.health()
            return health["status"] in ["green", "yellow"]
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        try:
            stats = self.client.indices.stats(index=self.index_name)
            count = self.client.count(index=self.index_name)

            return {
                "index_name": self.index_name,
                "document_count": count["count"],
                "size_in_bytes": stats["indices"][self.index_name]["total"]["store"]["size_in_bytes"],
                "health": self.client.cluster.health(index=self.index_name)["status"],
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"error": str(e)}

    def get_cluster_info(self) -> Optional[Dict[str, Any]]:
        """Get OpenSearch cluster information."""
        try:
            return self.client.info()
        except Exception as e:
            logger.error(f"Error getting cluster info: {e}")
            return None

    def get_index_mapping(self) -> Optional[Dict[str, Any]]:
        """Get index mapping."""
        try:
            mappings = self.client.indices.get_mapping(index=self.index_name)
            if mappings and self.index_name in mappings:
                return mappings[self.index_name].get("mappings", {})
            return {}
        except Exception as e:
            logger.error(f"Error getting index mapping: {e}")
            return None