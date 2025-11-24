from __future__ import annotations
import numpy as np
from typing import Callable, List, Sequence, Tuple, Optional, Dict, Any, Iterator
from src.mlp.activations import Activation
from src.mlp.erorrs import Loss
from src.mlp.optimizers import Optimizer

# ---------------------------
# Layer
# ---------------------------
class Layer:
    """
    W: (n_out, n_in+1)  [col 0 = bias]
    forward(a_prev) recibe vector sin bias y devuelve activación sin bias.
    """
    def __init__(self, n_inputs: int, n_neurons: int, activation: Activation,
                 w_init_scale: float = 0.05, rng: Optional[np.random.Generator] = None):
        self.activation = activation
        self.rng = rng or np.random.default_rng()
        self.weights = self.rng.uniform(-w_init_scale, w_init_scale, size=(n_neurons, n_inputs+1))
        self.z: Optional[np.ndarray] = None
        self.a: Optional[np.ndarray] = None
    @property
    def shape(self): return self.weights.shape
    def forward(self, a_prev: np.ndarray) -> np.ndarray:
        a_prev_b = np.concatenate(([1.0], a_prev))
        self.z = self.weights @ a_prev_b
        self.a = self.activation.f(self.z)
        return self.a
    def backprop_delta(self, delta_next: np.ndarray, weights_next: np.ndarray) -> np.ndarray:
        Wn = weights_next[:, 1:]  # ignora columna de bias de la siguiente capa
        return (Wn.T @ delta_next) * self.activation.df(self.z, self.a)  # type: ignore
    def grad_w(self, a_prev: np.ndarray, delta: np.ndarray) -> np.ndarray:
        a_prev_b = np.concatenate(([1.0], a_prev))
        return np.outer(delta, a_prev_b)

# ---------------------------
# MLP
# ---------------------------
class MLP:
    """
    layer_sizes: [n_in, h1, ..., hK, n_out]
    activations: [act(h1), ..., act(hK), act(out)]
    """
    def __init__(self,
                 layer_sizes: Sequence[int],
                 activations: Sequence[Activation],
                 loss: Loss,
                 optimizer: Optimizer,
                 w_init_scale: float = 0.05,
                 seed: Optional[int] = None):
        assert len(layer_sizes) >= 2
        assert len(activations) == len(layer_sizes) - 1
        rng = np.random.default_rng(seed)
        self._rng = rng  # <- para barajar reproducible
        self.layers: List[Layer] = [
            Layer(layer_sizes[i-1], layer_sizes[i], activations[i-1], w_init_scale, rng)
            for i in range(1, len(layer_sizes))
        ]
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer.init([ly.shape for ly in self.layers])
        self.error_history: list[float] = []


    def _forward_full(self, x: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        a = x
        zs: List[np.ndarray] = []
        acts: List[np.ndarray] = [a]
        for layer in self.layers:
            a = layer.forward(a)
            zs.append(layer.z.copy())      # type: ignore
            acts.append(a.copy())
        return zs, acts

    def forward(self, x: np.ndarray) -> np.ndarray:
        _, acts = self._forward_full(x)
        return acts[-1]

    def _backward_update(self, x: np.ndarray, y: np.ndarray) -> float:
        zs, acts = self._forward_full(x)
        y_hat = acts[-1]
        delta = self.loss.delta_out(y_hat, y, zs[-1], self.layers[-1].activation)

        deltas: List[np.ndarray] = [None] * len(self.layers)  # type: ignore
        deltas[-1] = delta
        for l in reversed(range(len(self.layers)-1)):
            deltas[l] = self.layers[l].backprop_delta(deltas[l+1], self.layers[l+1].weights)

        self.optimizer.begin_step()
        for l, layer in enumerate(self.layers):
            g = layer.grad_w(acts[l], deltas[l])
            step = self.optimizer.update(l, g)
            layer.weights -= step   # regla consistente (W -= lr * grad-like)
        return self.loss.value(y_hat, y)

    def backprop_to_input(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Backpropagate to get gradient at input layer without updating weights.
        
        Args:
            x: Input data (input_dim,) or (n_samples, input_dim)
            y: Target data (output_dim,) or (n_samples, output_dim)
            
        Returns:
            Gradient at input: ∂L/∂x (same shape as x)
        """
        zs, acts = self._forward_full(x)
        y_hat = acts[-1]
        delta = self.loss.delta_out(y_hat, y, zs[-1], self.layers[-1].activation)
        
        deltas: List[np.ndarray] = [None] * len(self.layers)  # type: ignore
        deltas[-1] = delta
        for l in reversed(range(len(self.layers)-1)):
            deltas[l] = self.layers[l].backprop_delta(deltas[l+1], self.layers[l+1].weights)
        
        # Get gradient at input: backprop through first layer
        if len(self.layers) > 0:
            # Gradient w.r.t. input = W^T @ delta (without activation derivative at input)
            W0 = self.layers[0].weights[:, 1:]  # Remove bias column
            delta_input = W0.T @ deltas[0]
            return delta_input
        return delta

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int = 100,
            batch_size: int = 1, shuffle: bool = True, verbose: bool = False,
            validation_fn: Optional[Callable[[np.ndarray, np.ndarray, 'MLP'], float]] = None,
            max_validation_error: Optional[float] = None,
            check_every: int = 1) -> List[float]:
        """
        Entrena el MLP.
        
        Args:
            X: Datos de entrada (N, input_dim)
            Y: Datos objetivo (N, output_dim)
            epochs: Número máximo de épocas
            batch_size: Tamaño del batch
            shuffle: Si barajar los datos
            verbose: Si imprimir progreso
            validation_fn: Función que calcula un error de validación: validation_fn(X, Y, model) -> error
                         Se usa para early stopping si max_validation_error no es None.
            max_validation_error: Si se proporciona y validation_fn existe, detiene el entrenamiento
                                cuando validation_fn(X, Y, model) <= max_validation_error.
            check_every: Cada cuántas épocas verificar el error de validación (default: 1 = cada época)
        
        Returns:
            Historial de pérdidas
        """
        N = X.shape[0]; hist: List[float] = []
        for ep in range(epochs):
            idx = np.arange(N)
            if shuffle: self._rng.shuffle(idx)  # <- usa el RNG de la clase
            epoch_loss = 0.0

            if batch_size <= 1:
                for i in idx:
                    epoch_loss += self._backward_update(X[i], Y[i])
            else:
                for s in range(0, N, batch_size):
                    e = min(s + batch_size, N)
                    bidx = idx[s:e]
                    acts_list: List[List[np.ndarray]] = []
                    deltas_list: List[List[np.ndarray]] = []
                    b_loss = 0.0
                    for i in bidx:
                        zs, acts = self._forward_full(X[i]); y_hat = acts[-1]
                        d_out = self.loss.delta_out(y_hat, Y[i], zs[-1], self.layers[-1].activation)
                        deltas = [None] * len(self.layers)  # type: ignore
                        deltas[-1] = d_out
                        for l in reversed(range(len(self.layers)-1)):
                            deltas[l] = self.layers[l].backprop_delta(deltas[l+1], self.layers[l+1].weights)
                        acts_list.append(acts); deltas_list.append(deltas)
                        b_loss += self.loss.value(y_hat, Y[i])

                    self.optimizer.begin_step()
                    for l, layer in enumerate(self.layers):
                        G = np.zeros_like(layer.weights)
                        for acts, deltas in zip(acts_list, deltas_list):
                            G += layer.grad_w(acts[l], deltas[l])
                        G /= len(bidx)
                        step = self.optimizer.update(l, G)
                        layer.weights -= step
                    epoch_loss += b_loss

            epoch_loss /= N; hist.append(epoch_loss)
            
            # Verificar early stopping basado en error de validación
            should_check = (validation_fn is not None and max_validation_error is not None and
                          ((ep + 1) % check_every == 0 or ep == 0))
            
            if should_check:
                validation_error = validation_fn(X, Y, self)
                should_stop = validation_error <= max_validation_error
                
                if verbose or should_stop:
                    msg = f"[{ep+1:03d}/{epochs}] loss={epoch_loss:.6f}, validation_error={validation_error:.2f}"
                    print(msg)
                
                if should_stop:
                    print(f"\n✓ Training stopped early: validation error {validation_error:.2f} <= {max_validation_error}")
                    break
            elif verbose:
                print(f"[{ep+1:03d}/{epochs}] loss={epoch_loss:.6f}")
        
        self.error_history = hist.copy()
        return hist

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            return self.forward(X)
        if X.shape[0] == 0:  # <- protección: batch vacío
            out_dim = self.layers[-1].weights.shape[0]
            return np.empty((0, out_dim))
        return np.vstack([self.forward(x) for x in X])

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        y = self.predict(X)
        if y.ndim == 1 or y.shape[1] == 1:
            return (y > 0.5).astype(int).reshape(-1,1)
        return np.argmax(y, axis=1).reshape(-1,1)
