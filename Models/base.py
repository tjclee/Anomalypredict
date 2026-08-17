from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class AnomalyDetector(ABC):
    @abstractmethod
    def fit(self, windows: np.ndarray) -> "AnomalyDetector":
        raise NotImplementedError
    
    @abstractmethod
    def score(self, windows: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @staticmethod
    def _flatten(windows: np.ndarray) -> np.ndarray:
        return windows.reshape(windows.shape[0], -1)

