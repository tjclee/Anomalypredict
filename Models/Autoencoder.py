from __future__ import annotations
import numpy as np
from .base import AnomalyDetector

class DenseAutoencoderDetector(AnomalyDetector):
    def __init__(
        self,
        latent_dim: int = 8,
        hidden_dims: tuple[int, ...] = (64, 32),
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        l2_reg: float = 0.0,
        random_state: int | None = None,
        verbose: bool = False,
    ):

        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if any(h < 1 for h in hidden_dims):
            raise ValueError(f"hidden_dims must all be >= 1, got {hidden_dims}")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if batch_size < 1:
            raise ValueError(f"batch_size msut be >= 1, got {batch_size}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        
        self.latent_dim = latent_dim
        self.hidden_dims = tuple(hidden_dims)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.random_state = random_state
        self.verbose = verbose

        self.layer_dims_: list[int] | None = None
        self.weights_: list[np.ndarray] = []
        self.biases_: list[np.ndarray] = []
        self.loss_history_: list[float] = []

        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None

    def _build_layer_dims(self, input_dim: int) -> list[int]:
        encoder = [input_dim, *self.hidden_dims, self.latent_dim]
        decoder = [self.latent_dim, *reversed(self.hidden_dims), input_dim]
        return encoder + decoder[1:]
    
    def _init_weights(self, rng: np.random.Generator) -> None:
        self.weights_ = []
        self.biases_ = []
        dims = self.layer_dims_
        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i+1]
            limit = np.sqrt(2.0 / fan_in)
            W = rng.normal(0.0, limit, size=(fan_in, fan_out))
            b = np.zeros(fan_out)
            self.weights_.append(W)
            self.biases_.append(b)


    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)
    
    @staticmethod
    def _relu_grad(z: np.ndarray) -> np.ndarray:
        return (z > 0.0).astype(z.dtype)
    
    def _forward(self, X: np.ndarray):
        n_layers = len(self.weights_)
        activations = [X]
        pre_activations = []
        a = X
        for i in range(n_layers):
            z = a @ self.weights_[i] + self.biases_[i]
            pre_activations.append(z)
            is_output_layer = (i == n_layers - 1)
            a = z if is_output_layer else self._relu(z)
            activations.append(a)
        return activations, pre_activations
    
    
        