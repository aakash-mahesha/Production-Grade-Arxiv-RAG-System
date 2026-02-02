ARXIV_PAPERS_INDEX = "arxiv-papers"

# Index mapping configuration for arXiv papers
ARXIV_PAPERS_MAPPING = {
    "settings": {
        "number_of_shards": 1,      # Single shard for small dataset
        "number_of_replicas": 0,    # No replicas in dev
        "analysis": {
            "analyzer": {
                # Standard analyzer with English stopwords
                "standard_analyzer": {
                    "type": "standard",
                    "stopwords": "_english_"
                },
                # Custom text analyzer with stemming
                "text_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "stop", "snowball"]
                }
            }
        }
    },
    "mappings": {
        "dynamic": "strict",  # Reject unknown fields
        "properties": {
            # Exact match field (for filtering)
            "arxiv_id": {"type": "keyword"},
            
            # Full-text searchable with keyword subfield
            "title": {
                "type": "text",
                "analyzer": "text_analyzer",
                "fields": {
                    "keyword": {"type": "keyword", "ignore_above": 256}
                }
            },
            
            # Authors - searchable text
            "authors": {
                "type": "text",
                "analyzer": "standard_analyzer",
                "fields": {
                    "keyword": {"type": "keyword", "ignore_above": 256}
                }
            },
            
            # Abstract - main search content
            "abstract": {
                "type": "text",
                "analyzer": "text_analyzer"
            },
            
            # Categories - exact match filtering
            "categories": {"type": "keyword"},
            
            # Full text from PDF (if available)
            "raw_text": {
                "type": "text",
                "analyzer": "text_analyzer"
            },
            
            # URL - not analyzed
            "pdf_url": {"type": "keyword"},
            
            # Date fields
            "published_date": {"type": "date"},
            "created_at": {"type": "date"},
            "updated_at": {"type": "date"},
        }
    }
}