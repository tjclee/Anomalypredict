import io

import numpy as np
import pandas as pd
import pytest

from FileHandler.exceptions import EmptyDatasetError, NoNumericColumnsError
from FileHandler.loader import TimeSeriesIngestor


def _csv(text: str) -> io.StringIO:
    return io.StringIO(text.strip() + "\n")


def test_clean_csv_with_timestamp():
    csv = _csv(
        """
        timestamp,sensor_a,sensor_b
        2024-01-01 00:00:00,1.0,10.0
        2024-01-01 00:01:00,1.1,10.5
        2024-01-01 00:02:00,1.2,11.0
        """
    )
    result = TimeSeriesIngestor().load(csv)
    assert result.report.timestamp_column == "timestamp"
    assert list(result.data.columns) == ["sensor_a", "sensor_b"]
    assert result.report.sampling_is_regular is True
    assert result.report.inferred_sampling_seconds == 60.0
    assert len(result.data) == 3


def test_no_timestamp_column_falls_back_to_range_index():
    csv = _csv(
        """
        sensor_a,sensor_b
        1.0,10.0
        1.1,10.5
        1.2,11.0
        """
    )
    result = TimeSeriesIngestor().load(csv)
    assert result.report.timestamp_column is None
    assert any("no timestamp" in w.lower() for w in result.report.warnings)
    assert len(result.data) == 3


def test_non_numeric_junk_column_is_dropped():
    csv = _csv(
        """
        timestamp,sensor_a,notes
        2024-01-01 00:00:00,1.0,fine
        2024-01-01 00:01:00,1.1,a bit noisy today
        2024-01-01 00:02:00,1.2,looks ok
        """
    )
    result = TimeSeriesIngestor().load(csv)
    assert "notes" in result.report.columns.dropped_non_numeric
    assert "sensor_a" in result.data.columns


def test_currency_formatted_numbers_are_cleaned():
    csv = _csv(
        """
        timestamp,revenue
        2024-01-01,"$1,000.50"
        2024-01-02,"$1,200.00"
        2024-01-03,"$1,150.25"
        """
    )
    result = TimeSeriesIngestor().load(csv)
    assert result.data["revenue"].iloc[0] == pytest.approx(1000.50)


def test_constant_column_is_dropped():
    csv = _csv(
        """
        timestamp,sensor_a,mode
        2024-01-01 00:00:00,1.0,5
        2024-01-01 00:01:00,1.1,5
        2024-01-01 00:02:00,1.2,5
        """
    )
    result = TimeSeriesIngestor().load(csv)
    assert "mode" in result.report.columns.dropped_constant
    assert "mode" not in result.data.columns


def test_all_nan_column_is_dropped():
    csv = _csv(
        """
        timestamp,sensor_a,broken
        2024-01-01 00:00:00,1.0,
        2024-01-01 00:01:00,1.1,
        2024-01-01 00:02:00,1.2,
        """
    )
    result = TimeSeriesIngestor().load(csv)
    assert "broken" in result.report.columns.dropped_all_nan


def test_missing_values_are_interpolated_by_default():
    csv = _csv(
        """
        timestamp,sensor_a
        2024-01-01 00:00:00,1.0
        2024-01-01 00:01:00,
        2024-01-01 00:02:00,3.0
        """
    )
    result = TimeSeriesIngestor().load(csv)
    assert result.data["sensor_a"].iloc[1] == pytest.approx(2.0)
    assert result.report.missing_value_fraction > 0


def test_missing_values_drop_rows_strategy():
    csv = _csv(
        """
        timestamp,sensor_a
        2024-01-01 00:00:00,1.0
        2024-01-01 00:01:00,
        2024-01-01 00:02:00,3.0
        """
    )
    result = TimeSeriesIngestor(missing_value_strategy="drop_rows").load(csv)
    assert len(result.data) == 2


def test_duplicate_timestamps_are_dropped_keeping_first():
    csv = _csv(
        """
        timestamp,sensor_a
        2024-01-01 00:00:00,1.0
        2024-01-01 00:00:00,999.0
        2024-01-01 00:01:00,1.1
        """
    )
    result = TimeSeriesIngestor().load(csv)
    assert result.report.n_duplicate_timestamps_dropped == 1
    assert result.data["sensor_a"].iloc[0] == 1.0


def test_unsorted_rows_are_resorted():
    csv = _csv(
        """
        timestamp,sensor_a
        2024-01-01 00:02:00,3.0
        2024-01-01 00:00:00,1.0
        2024-01-01 00:01:00,2.0
        """
    )
    result = TimeSeriesIngestor().load(csv)
    assert result.report.was_unsorted is True
    assert list(result.data["sensor_a"]) == [1.0, 2.0, 3.0]


def test_irregular_sampling_is_flagged():
    csv = _csv(
        """
        timestamp,sensor_a
        2024-01-01 00:00:00,1.0
        2024-01-01 00:01:00,1.1
        2024-01-01 00:45:00,1.2
        """
    )
    result = TimeSeriesIngestor().load(csv)
    assert result.report.sampling_is_regular is False
    assert any("unevenly spaced" in w.lower() for w in result.report.warnings)


def test_empty_file_raises():
    csv = _csv("timestamp,sensor_a\n")
    with pytest.raises(EmptyDatasetError):
        TimeSeriesIngestor().load(csv)


def test_all_columns_unusable_raises():
    csv = _csv(
        """
        timestamp,notes,id
        2024-01-01 00:00:00,hello there,abc-123
        2024-01-01 00:01:00,goodbye now,def-456
        2024-01-01 00:02:00,another note,ghi-789
        """
    )
    with pytest.raises(NoNumericColumnsError):
        TimeSeriesIngestor().load(csv)


def test_min_numeric_columns_threshold():
    csv = _csv(
        """
        timestamp,sensor_a
        2024-01-01 00:00:00,1.0
        2024-01-01 00:01:00,1.1
        """
    )
    with pytest.raises(NoNumericColumnsError):
        TimeSeriesIngestor(min_numeric_columns=2).load(csv)


def test_timestamp_like_name_but_unparseable_is_kept_as_regular_column():
    csv = _csv(
        """
        date,sensor_a
        not-a-date,1.0
        also-not,1.1
        nope,1.2
        """
    )
    result = TimeSeriesIngestor().load(csv)
    assert result.report.timestamp_column is None
    assert any("looks like a timestamp" in w for w in result.report.warnings)