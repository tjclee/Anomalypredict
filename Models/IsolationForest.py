from __future__ import annotations
import math
import numpy as np
from .base import AnomalyDetector

class _Node:
    __slots__ = ("feature", "split_value", "left", "right", "size")

    def __init__(self, feature=None, split_value = None, left = None, right = None, size = 0):
        self.feature = feature
        self.split_value = split_value
        self.left = left
        self.right = right
        self.size = size

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None
    
def _average_path_length(n: int) -> float:
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    harmonic = math.log(n-1) + 0.5772156649015329
    return 2.0 * harmonic - (2.0 * (n-1) / n)

class IsolationTree:
    def __init__(self, max_depth: int, random_state: np.random.Generator):
        self.max_depth = max_depth
        self.rng = random_state
        self.root: _Node | None = None

    def fit(self, X: np.ndarray) -> "IsolationTree":
        self.root = self._build(X, depth=0)
        return self
    
    def _build(self, X: np.ndarray, depth: int) -> _Node:
        n_samples, n_features = X.shape


        if depth >= self.max_depth or n_samples <= 1:
            return _Node(size=n_samples)
        
        feature = self.rng.integers(0, n_features)
        col = X[:, feature]
        col_min, col_max = col.min(), col.max()

        if col_min == col_max:
            return _Node(size=n_samples)
        
        split_value = self.rng.uniform(col_min, col_max)

        left_mask = col < split_value
        left = self._build(X[left_mask], depth + 1)
        right = self._build(X[~left_mask], depth + 1)
        return _Node(feature = feature, split_value=split_value, left=left, right=right)
    
    def path_length(self, x: np.ndarray) -> float:
        node = self.root
        depth = 0
        while not node.is_leaf:
            if x[node.feature] < node.split_value:
                node = node.left
            else:
                node = node.right
            depth +=1
        return depth + _average_path_length(node.size)

class IsolationForestDetector(AnomalyDetector):
    def __init__(self, n_estimators: int = 100, max_samples: int = 256, max_depth: int | None = None, random_state: int | None = None):
        if n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {n_estimators}")
        if max_samples < 2:
            raise ValueError(f"max_samples msut be >= 2, got {max_samples}")
        
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.random_state = random_state

        self.trees_: list[IsolationTree] = []
        self._effective_max_samples: int | None = None

    def fit(self, windows: np.ndarray) -> "IsolationForestDetector":
        X = self._flatten(windows).astype(np.float64)
        n_points = X.shape[0]

        sample_size = min(self.max_samples, n_points)
        self._effective_max_samples = sample_size
        depth_limit = self.max_depth or math.ceil(math.log2(max(sample_size, 2)))

        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        for _ in range(self.n_estimators):
            idx= rng.choice(n_points, size=sample_size, replace=False)
            tree = IsolationTree(max_depth=depth_limit, random_state=rng).fit(X[idx])
            self.trees_.append(tree)
        return self
        
    def score(self, windows: np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise RuntimeError("Call .fit() before .score().")
            
        X = self._flatten(windows).astype(np.float64)
        n = self._effective_max_samples
        c_n = _average_path_length(n)

        avg_path = np.array(
            [np.mean([tree.path_length(x) for tree in self.trees_]) for x in X]
        )

        return np.power(2.0, -avg_path / c_n)

