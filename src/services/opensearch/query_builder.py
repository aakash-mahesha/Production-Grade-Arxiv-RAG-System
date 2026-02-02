import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PaperQueryBuilder:
    """
    Query builder for arXiv papers search.
    
    Builds OpenSearch queries with:
    - Multi-field search with boosting
    - BM25 relevance scoring
    - Category filtering
    - Result highlighting
    """

    def __init__(
        self,
        query: str,
        size: int = 10,
        from_: int = 0,
        fields: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        track_total_hits: bool = True,
        latest_papers: bool = False,
    ):
        """
        Initialize query builder.

        Args:
            query: Search query text
            size: Number of results to return
            from_: Offset for pagination
            fields: Fields to search (with optional boosting)
            categories: Filter by categories
            track_total_hits: Whether to track total hits accurately
            latest_papers: Sort by date instead of relevance
        """
        self.query = query
        self.size = size
        self.from_ = from_
        # Field boosting: title (3x) > abstract (2x) > authors (1x)
        self.fields = fields or ["title^3", "abstract^2", "authors^1"]
        self.categories = categories
        self.track_total_hits = track_total_hits
        self.latest_papers = latest_papers

    def build(self) -> Dict[str, Any]:
        """Build the complete OpenSearch query."""
        query_body = {
            "query": self._build_query(),
            "size": self.size,
            "from": self.from_,
            "track_total_hits": self.track_total_hits,
            "_source": self._build_source_fields(),
            "highlight": self._build_highlight(),
        }

        # Add sorting if needed
        sort = self._build_sort()
        if sort:
            query_body["sort"] = sort

        return query_body

    def _build_query(self) -> Dict[str, Any]:
        """Build the main query with filters."""
        must_clauses = []

        # Main text search
        if self.query.strip():
            must_clauses.append(self._build_text_query())

        # Build filter clauses
        filter_clauses = self._build_filters()

        # Construct bool query
        bool_query = {}

        if must_clauses:
            bool_query["must"] = must_clauses
        else:
            # If no text query, match all documents
            bool_query["must"] = [{"match_all": {}}]

        if filter_clauses:
            bool_query["filter"] = filter_clauses

        return {"bool": bool_query}

    def _build_text_query(self) -> Dict[str, Any]:
        """Build the multi-match text search query."""
        return {
            "multi_match": {
                "query": self.query,
                "fields": self.fields,       # title^3, abstract^2, authors^1
                "type": "best_fields",       # Use highest scoring field
                "operator": "or",            # Any term can match
                "fuzziness": "AUTO",         # Handle typos
                "prefix_length": 2,          # Min chars before fuzzy kicks in
            }
        }

    def _build_filters(self) -> List[Dict[str, Any]]:
        """Build filter clauses (don't affect scoring)."""
        filters = []

        if self.categories:
            filters.append({"terms": {"categories": self.categories}})

        return filters

    def _build_source_fields(self) -> List[str]:
        """Define which fields to return in results."""
        return [
            "arxiv_id",
            "title",
            "authors",
            "abstract",
            "categories",
            "published_date",
            "pdf_url"
        ]

    def _build_highlight(self) -> Dict[str, Any]:
        """Build highlighting configuration."""
        return {
            "fields": {
                "title": {
                    "fragment_size": 0,        # Return entire field
                    "number_of_fragments": 0,
                },
                "abstract": {
                    "fragment_size": 150,      # 150 char snippets
                    "number_of_fragments": 3,  # Up to 3 snippets
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"],
                },
                "authors": {
                    "fragment_size": 0,
                    "number_of_fragments": 0,
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"],
                },
            },
            "require_field_match": False,
        }

    def _build_sort(self) -> Optional[List[Dict[str, Any]]]:
        """Build sorting configuration."""
        # Sort by date if requested
        if self.latest_papers:
            return [{"published_date": {"order": "desc"}}, "_score"]

        # For text queries, use relevance scoring (no explicit sort)
        if self.query.strip():
            return None

        # For empty queries, sort by date
        return [{"published_date": {"order": "desc"}}, "_score"]


def build_search_query(
    query: str,
    size: int = 10,
    from_: int = 0,
    categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Helper function to build a search query."""
    builder = PaperQueryBuilder(
        query=query,
        size=size,
        from_=from_,
        categories=categories
    )
    return builder.build()