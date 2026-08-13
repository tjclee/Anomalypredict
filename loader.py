from __future__ import annotations

import io
import warnings as _warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .exceptions import EmptyDatasetError, NoNumericColumnsError


_TIMESTEP_NAME_CANDIDATES = (
    "timestamp", "time", "datetime", "date", "ds", "t",
)


SourceType = Union[str, Path, io.IOBase]

@dataclass
class ColumnReport:
    kept: List[str] = field(default_factory=list)
    dropped_non_numeric: List[str] = field(default_factory=list)
    dropped_constant: List[str] = field(default_factory=list)
    dropped_all_nan: List[str] = field(default_factory=list)

@dataclass
class IngestionReport:
    n_rows_raw: int
    n_rows_clean: int
    n_duplicate_timestamps_dropped: int
    was_unsorted:bool
    timestamp_column: Optional[str]
    inferred_sampling_seconds: Optional[float]
    sampling_is_regular: Optional[bool]
    missing_value_fraction: float
    columns: ColumnReport
    warnings: List[str] = field(default_factory=list)