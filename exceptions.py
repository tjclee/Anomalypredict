class IngestionError(Exception):
    """Base class for all ingestion-related failures"""

class EmptyDatasetError(IngestionError):
    """Raised when the uploaded file has no rows after parsing"""

class NoNumericColumnsError(IngestionError):
    """Raised when, after cleaning, zero usable numeric columsn remain"""

class InsufficientDataError(IngestionError):
    """Raised when there are fewer rows than the requested window size."""


