from __future__ import annotations
import numpy as np
from .base import AnomalyDetector

class PCAReconstructionDetector(AnomalyDetector):
    def __init__(self, n_components: int = 2):
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None


    def fit(self, windows: np.ndarray) -> PCAReconstructionDetector:
        X = self._flatten(windows).astype(np.float64)
        n_windows, n_dims = X.shape
        if self.n_components > min(n_windows, n_dims):
            raise ValueError(
                f"n_components={self.n_components} exceeds min(n_windows={n_windows}, "
                f"n_dims={n_dims}) -- cannot extract that many directions."
            )
        
        #Center the data
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        #SVD Decompose
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)

        #Keep top n components directions
        self.components_ = Vt[: self.n_components]
        return self
    def score(self, windows: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("Call .fit() before .score().")

        X = self._flatten(windows).astype(np.float64)
        X_centered = X - self.mean_
        projected = X_centered @ self.components_.T
        reconstructed = projected @ self.components_

        error = ((X_centered - reconstructed) ** 2).mean(axis=1)
        return error
    
    def explained_variance_ratio(self, windows: np.ndarray) -> float:
        X = self._flatten(windows).astype(np.float64)
        X_centered = X - self.mean_
        total_variance = (X_centered ** 2).sum()
        projected = X_centered @ self.components_.T
        captured_variance = (projected ** 2).sum()
        return float(captured_variance / total_variance)