from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import pandas as pd

from .exceptions import InsufficientDataError

def sliding_windows(
        data: Union[pd.DataFrame, np.ndarray],
        window_size: int,
        stride: int = 1,     
) -> Tuple[np.ndarray, np.ndarray]:
    if window_size < 1:
        raise ValueError(f"window size must be >= 1, got {window_size}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    
    values = data.values if isinstance(data, pd.DataFrame) else np.asarray(data)
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    n_rows = values.shape[0]
    if n_rows < window_size:
        raise InsufficientDataError(
            f"Need at least {window_size} rows to form one window, got {n_rows}."
        )
    

    starts = np.arange(0, n_rows - window_size + 1, stride)
    n_features = values.shape[1]
    item_size = values.strides[-1]
    windows = np.lib.stride_tricks.as_strided(
        values,
        shape=(len(starts), window_size, n_features),
        strides=(stride * values.strides[0], values.strides[0], item_size),
        writeable=False,
    )
    end_indices = starts + window_size - 1
    return windows, end_indices