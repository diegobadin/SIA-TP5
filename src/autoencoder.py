from __future__ import annotations
from typing import List, Sequence, Optional, Tuple

import numpy as np

from src.mlp.mlp import MLP
from src.mlp.activations import Activation
from src.mlp.erorrs import Loss
from src.mlp.optimizers import Optimizer


class Encoder:
    """Codificador que reduce la dimensionalidad de entrada a un espacio latente."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 hidden_layers: Sequence[int],
                 activations: Sequence[Activation],
                 loss: Loss,
                 optimizer: Optimizer,
                 w_init_scale: float = 0.05,
                 seed: Optional[int] = None):
        layer_sizes = [input_dim] + list(hidden_layers) + [latent_dim]
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError(f"Se necesitan {len(layer_sizes) - 1} activaciones, se proporcionaron {len(activations)}")

        self.mlp = MLP(
            layer_sizes=layer_sizes,
            activations=activations,
            loss=loss,
            optimizer=optimizer,
            w_init_scale=w_init_scale,
            seed=seed
        )

    def encode(self, x: np.ndarray) -> np.ndarray:
        return self.mlp.forward(x)

    def get_weights(self) -> List[np.ndarray]:
        return [layer.weights.copy() for layer in self.mlp.layers]

    def set_weights(self, weights: List[np.ndarray]):
        if len(weights) != len(self.mlp.layers):
            raise ValueError(f"Se esperaban {len(self.mlp.layers)} capas, se proporcionaron {len(weights)}")
        for layer, w in zip(self.mlp.layers, weights):
            layer.weights = w.copy()


class Decoder:
    """Decodificador que reconstruye la entrada desde el espacio latente."""
    def __init__(self,
                 latent_dim: int,
                 output_dim: int,
                 hidden_layers: Sequence[int],
                 activations: Sequence[Activation],
                 loss: Loss,
                 optimizer: Optimizer,
                 w_init_scale: float = 0.05,
                 seed: Optional[int] = None):
        layer_sizes = [latent_dim] + list(hidden_layers) + [output_dim]
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError(f"Se necesitan {len(layer_sizes) - 1} activaciones, se proporcionaron {len(activations)}")

        self.mlp = MLP(
            layer_sizes=layer_sizes,
            activations=activations,
            loss=loss,
            optimizer=optimizer,
            w_init_scale=w_init_scale,
            seed=seed
        )

    def decode(self, z: np.ndarray) -> np.ndarray:
        return self.mlp.forward(z)

    def get_weights(self) -> List[np.ndarray]:
        return [layer.weights.copy() for layer in self.mlp.layers]

    def set_weights(self, weights: List[np.ndarray]):
        if len(weights) != len(self.mlp.layers):
            raise ValueError(f"Se esperaban {len(self.mlp.layers)} capas, se proporcionaron {len(weights)}")
        for layer, w in zip(self.mlp.layers, weights):
            layer.weights = w.copy()


class Autoencoder:
    """Autoencoder que combina Encoder y Decoder y entrena end-to-end."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 encoder_hidden: Sequence[int],
                 decoder_hidden: Sequence[int],
                 encoder_activations: Sequence[Activation],
                 decoder_activations: Sequence[Activation],
                 loss: Loss,
                 optimizer: Optimizer,
                 w_init_scale: float = 0.05,
                 seed: Optional[int] = None):

        self.encoder = Encoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_layers=encoder_hidden,
            activations=encoder_activations,
            loss=loss,
            optimizer=optimizer,
            w_init_scale=w_init_scale,
            seed=seed
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_layers=decoder_hidden,
            activations=decoder_activations,
            loss=loss,
            optimizer=optimizer,
            w_init_scale=w_init_scale,
            seed=seed
        )

        full_layer_sizes = ([input_dim] + list(encoder_hidden) + [latent_dim] +
                            list(decoder_hidden) + [input_dim])
        full_activations = list(encoder_activations) + list(decoder_activations)

        self.mlp = MLP(
            layer_sizes=full_layer_sizes,
            activations=full_activations,
            loss=loss,
            optimizer=optimizer,
            w_init_scale=w_init_scale,
            seed=seed
        )
        self._sync_weights_from_mlp()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.loss = loss
        self.optimizer = optimizer

    def _sync_weights_from_mlp(self):
        num_encoder_layers = len(self.encoder.mlp.layers)
        num_decoder_layers = len(self.decoder.mlp.layers)

        for i in range(num_encoder_layers):
            self.encoder.mlp.layers[i].weights = self.mlp.layers[i].weights.copy()

        decoder_start_idx = num_encoder_layers
        for i in range(num_decoder_layers):
            self.decoder.mlp.layers[i].weights = self.mlp.layers[decoder_start_idx + i].weights.copy()

    def _sync_weights_to_mlp(self):
        num_encoder_layers = len(self.encoder.mlp.layers)
        num_decoder_layers = len(self.decoder.mlp.layers)

        for i in range(num_encoder_layers):
            self.mlp.layers[i].weights = self.encoder.mlp.layers[i].weights.copy()

        decoder_start_idx = num_encoder_layers
        for i in range(num_decoder_layers):
            self.mlp.layers[decoder_start_idx + i].weights = self.decoder.mlp.layers[i].weights.copy()

    def encode(self, x: np.ndarray) -> np.ndarray:
        num_encoder_layers = len(self.encoder.mlp.layers)
        a = x
        for i in range(num_encoder_layers):
            a = self.mlp.layers[i].forward(a)
        return a

    def decode(self, z: np.ndarray) -> np.ndarray:
        num_encoder_layers = len(self.encoder.mlp.layers)
        a = z
        for i in range(num_encoder_layers, len(self.mlp.layers)):
            a = self.mlp.layers[i].forward(a)
        return a

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z

    def fit(self, X_in: np.ndarray, X_target: np.ndarray,
            epochs: int = 100, batch_size: int = 1,
            shuffle: bool = True, verbose: bool = False) -> List[float]:
        """
        Entrena el autoencoder con entrada X_in y objetivo X_target (útil para denoising).
        """
        history = self.mlp.fit(X_in, X_target, epochs=epochs, batch_size=batch_size,
                               shuffle=shuffle, verbose=verbose)
        self._sync_weights_from_mlp()
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.mlp.predict(X)

    def get_latent_representation(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            return self.encode(X)
        return np.vstack([self.encode(x) for x in X])

    def reconstruct_error(self, X: np.ndarray) -> float:
        X_reconstructed = self.predict(X)
        return float(np.mean((X - X_reconstructed) ** 2))

    def pixel_error(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        X_reconstructed = self.predict(X)
        X_binary = (X > threshold).astype(float)
        X_recon_binary = (X_reconstructed > threshold).astype(float)
        return np.sum(X_binary != X_recon_binary, axis=1)

    def generate_from_latent(self, z: np.ndarray) -> np.ndarray:
        """
        Genera un nuevo carácter decodificando un punto en el espacio latente.
        
        Args:
            z: Vector latente de forma (latent_dim,) o (n_samples, latent_dim)
            
        Returns:
            Carácter(es) generado(s) de forma (output_dim,) o (n_samples, output_dim)
        """
        return self.decode(z)