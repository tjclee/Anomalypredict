import numpy as np
import pandas as pd
import pytest

from FileHandler.exceptions import InsufficientDataError
from FileHandler.windowing import sliding_windows


def test_basic_windowing_shape_and_values():
    data = np.arange(10).reshape(10, 1).astype(float)  # [[0],[1],...,[9]]
    windows, ends = sliding_windows(data, window_size=3, stride=1)
    assert windows.shape == (8, 3, 1)
    np.testing.assert_array_equal(windows[0].ravel(), [0, 1, 2])
    np.testing.assert_array_equal(windows[-1].ravel(), [7, 8, 9])
    np.testing.assert_array_equal(ends, np.arange(2, 10))


def test_stride_skips_windows():
    data = np.arange(10).reshape(10, 1).astype(float)
    windows, ends = sliding_windows(data, window_size=3, stride=3)
    # starts at 0, 3, 6 -> 3 windows (start=9 would need rows 9-11, out of range)
    assert windows.shape == (3, 3, 1)
    np.testing.assert_array_equal(ends, [2, 5, 8])


def test_multivariate_input():
    data = np.column_stack([np.arange(5), np.arange(5) * 10]).astype(float)
    windows, ends = sliding_windows(data, window_size=2, stride=1)
    assert windows.shape == (4, 2, 2)
    np.testing.assert_array_equal(windows[0], [[0, 0], [1, 10]])


def test_dataframe_input():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    windows, ends = sliding_windows(df, window_size=2, stride=1)
    assert windows.shape == (3, 2, 1)


def test_window_size_equal_to_data_length_gives_one_window():
    data = np.arange(5).reshape(5, 1).astype(float)
    windows, ends = sliding_windows(data, window_size=5, stride=1)
    assert windows.shape == (1, 5, 1)


def test_insufficient_data_raises():
    data = np.arange(3).reshape(3, 1).astype(float)
    with pytest.raises(InsufficientDataError):
        sliding_windows(data, window_size=5, stride=1)


def test_invalid_window_size_raises():
    data = np.arange(5).reshape(5, 1).astype(float)
    with pytest.raises(ValueError):
        sliding_windows(data, window_size=0, stride=1)


def test_invalid_stride_raises():
    data = np.arange(5).reshape(5, 1).astype(float)
    with pytest.raises(ValueError):
        sliding_windows(data, window_size=2, stride=0)


def test_windows_are_read_only_views():
    data = np.arange(10).reshape(10, 1).astype(float)
    windows, _ = sliding_windows(data, window_size=3, stride=1)
    with pytest.raises(ValueError):
        windows[0, 0, 0] = 999.0