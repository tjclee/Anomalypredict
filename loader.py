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

    def summary(self) -> str:
        lines = [
            f"{self.n_rows_clean}/{self.n_rows_raw} rows kept."
            f"{len(self.columns.kept)} numerical variable(s) kept"
        ]

        if self.columns.dropped_non_numeric:
            lines.append(
                f"Dropped non-numeric columns: {', '.join(self.columns.dropped_non_numeric)}"
            )

        if self.columns.dropped_constant:
            lines.append(
                f"Dropped constant (uninformative) columns: {', '.join(self.columns.dropped_constant)}"
            )

        if self.columns.dropped_all_nan:
            lines.append(
                f"Dropped empty columns: {', '.join(self.columns.dropped_all_nan)}"
            )

        if self.was_unsorted:
            lines.append(
                "Rows were not in time order; re-sorted by timestamp."
            )

        lines.extend(self.warnings)
        return "\n".join(lines)
    
@dataclass
class IngestionResult:
    data: pd.DataFrame
    report: IngestionReport

class TimeSeriesIngestor:
    def __init__(self,
                 min_numeric_columns: int = 1,
                 max_non_numeric_fraction: float = 0.1,
                 missing_value_strategy: str = "interpolate",
                 ):
        
        if missing_value_strategy not in ("interpolate", "drop_rows"):
            raise ValueError(
                f"missing_value_strategy msut be 'interpolate' or 'drop_rows', "
                f"got {missing_value_strategy}"
            )
        self.min_numeric_columns = min_numeric_columns
        self.max_non_numeric_fraction = max_non_numeric_fraction
        self.missing_value_strategy = missing_value_strategy

    def load(self, source: SourceType) -> IngestionResult:
        warnings: List[str] = []

        raw = self._read_csv(source)
        n_rows_raw = len(raw)
        if n_rows_raw == 0:
            raise EmptyDatasetError("Uploaded file contains no data rows")
        
        ts_col, raw = self._extract_timestamp(raw, warnings)
        if ts_col is None:
            warnings.append(
                "No timestamp column found; assuming rows are evenly-spaced, "
                "already ordered samples."
            )

        numeric_df, col_report = self._coerce_numeric_columns(
            raw.drop(columns=[ts_col]) if ts_col else raw, warnings
        )

        if len(col_report.kept) < self.min_numeric_columns:
            raise NoNumericColumnsError(
                f"Only {len(col_report.kept)} usable numeric column(s) found "
                f"(need >= {self.min_numeric_columns}). "
                f"Dropped: non-numeric={col_report.dropped_non_numeric}, "
                f"constant={col_report.dropped_constant}, "
                f"all-nan={col_report.dropped_all_nan}"
            )
        
        if ts_col:
            numeric_df.index = raw[ts_col]
            numeric_df.index.name = ts_col

        n_dupes = 0
        was_unsorted = False
        sampling_seconds = None
        sampling_regular = None

        if ts_col:
            numeric_df, n_dupes = self._drop_duplicate_timestamps(numeric_df)
            numeric_df, was_unsorted = self._sort_by_time(numeric_df)
            sampling_seconds, sampling_regular = self._infer_sampling(numeric_df.index)
            if sampling_regular is False:
                warnings.append(
                    "Timestamps are unevenly spaced; downstream windowing will "
                    "treat samples as sequential steps, not fixed wall-clock intervals."
                )

        missing_fraction = float(numeric_df.isna().mean().mean()) if len(numeric_df) else 0.0
        numeric_df = self._handle_missing(numeric_df)

        report = IngestionReport(
            n_rows_raw=n_rows_raw,
            n_rows_clean=len(numeric_df),
            n_duplicate_timestamps_dropped=n_dupes,
            was_unsorted=was_unsorted,
            timestamp_column=ts_col,
            inferred_sampling_seconds=sampling_seconds,
            sampling_is_regular=sampling_regular,
            missing_value_fraction=missing_fraction,
            columns=col_report,
            warnings=warnings,
        )

        return IngestionResult(data=numeric_df, report=report)
    
    @staticmethod
    def _read_csv(source: SourceType) -> pd.DataFrame:
        return pd.read_csv(source, skip_blank_lines=True)
    
    def _extract_timestamp(self, df: pd.DataFrame, warnings: list[str]):
        for col in df.columns:
            if col.strip().lower() in _TIMESTEP_NAME_CANDIDATES:
                parsed = self._safe_to_datetime(df[col])
                success = parsed.notna().mean()
                if success >= 0.9:
                    df = df.copy()
                    df[col] = parsed
                    return col, df
                warnings.append(
                    f"Column '{col}' looks like a timestamp by name but only "
                    f"{success:.0%} of values parsed; treating it as a regular column."
                )

        best_col, best_success, best_parsed = None, 0.0, None
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                parsed = self._safe_to_datetime(df[col])
                success = parsed.notna().mean()
                if success > best_success:
                    best_col, best_success, best_parsed = col, success, parsed
        if best_col is not None and best_success >= 0.9:
            df = df.copy()
            df[best_col] = best_parsed
            return best_col, df
        
        return None, df
    
    