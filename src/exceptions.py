class RepositoryException(Exception):
    """Base exception for repository-related errors."""


class PaperNotFound(RepositoryException):
    """Exception raised when paper data is not found."""


class PaperNotSaved(RepositoryException):
    """Exception raised when paper data is not saved."""


class ParsingException(Exception):
    """Base exception for parsing-related errors."""


class OpenSearchException(Exception):
    """Base exception for OpenSearch-related errors."""


class LLMException(Exception):
    """Base exception for LLM-related errors."""


class ConfigurationError(Exception):
    """Exception raised when configuration is invalid."""

# ArXiv API exceptions
class ArxivAPIException(Exception):
    """Base exception for arXiv API-related errors."""


class ArxivAPITimeoutError(ArxivAPIException):
    """Exception raised when arXiv API request times out."""


class ArxivAPIRateLimitError(ArxivAPIException):
    """Exception raised when arXiv API rate limit is exceeded."""


class ArxivParseError(ArxivAPIException):
    """Exception raised when arXiv API response parsing fails."""


# PDF download exceptions (used by ArXiv client)
class PDFDownloadException(Exception):
    """Base exception for PDF download-related errors."""


class PDFDownloadTimeoutError(PDFDownloadException):
    """Exception raised when PDF download times out."""

class ParsingException(Exception):
    """Base exception for parsing-related errors."""


class PDFParsingException(ParsingException):
    """Base exception for PDF parsing-related errors."""


class PDFValidationError(PDFParsingException):
    """Exception raised when PDF file validation fails."""


class MetadataFetchingException(Exception):
    """Base exception for metadata fetching pipeline errors."""


class PipelineException(MetadataFetchingException):
    """Exception raised during pipeline execution."""

