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

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int = 100,
            batch_size: int = 1, shuffle: bool = True, verbose: bool = False) -> List[float]:
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
            if verbose: print(f"[{ep+1:03d}/{epochs}] loss={epoch_loss:.6f}")
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

# ---------------------------
# Splits y métricas
# ---------------------------
def holdout_split(X: np.ndarray, Y: np.ndarray, test_size: float = 0.2,
                  shuffle: bool = True, seed: Optional[int] = None):
    n = len(X); idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed); rng.shuffle(idx)
    cut = int(round(n*(1-test_size)))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], X[te], Y[tr], Y[te]

class KFold:
    def __init__(self, n_splits: int = 5, shuffle: bool = True, seed: Optional[int] = None):
        assert n_splits >= 2
        self.k, self.shuffle, self.seed = n_splits, shuffle, seed
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Iterator[Tuple[np.ndarray,np.ndarray]]:
        n = len(X)
        k_eff = min(self.k, max(1, n))  # ajusta si n < k
        idx = np.arange(n)
        if self.shuffle:
            rng = np.random.default_rng(self.seed); rng.shuffle(idx)
        folds = np.array_split(idx, k_eff)
        for i in range(k_eff):
            val_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(k_eff) if j != i]) if k_eff > 1 else idx
            yield tr_idx, val_idx

class StratifiedKFold:
    def __init__(self, n_splits: int = 5, shuffle: bool = True, seed: Optional[int] = None):
        assert n_splits >= 2
        self.k, self.shuffle, self.seed = n_splits, shuffle, seed
    def _labels(self, Y: np.ndarray) -> np.ndarray:
        if Y.ndim == 2 and Y.shape[1] > 1: return np.argmax(Y, axis=1)
        return (Y.ravel() > 0.5).astype(int)
    def split(self, X: np.ndarray, Y: np.ndarray) -> Iterator[Tuple[np.ndarray,np.ndarray]]:
        n = len(X); lbl = self._labels(Y)
        rng = np.random.default_rng(self.seed)
        # k efectivo: no puede superar el tamaño de la clase minoritaria
        counts = [np.sum(lbl == c) for c in np.unique(lbl)]
        k_eff = min(self.k, max(1, min(counts)))

        per_class = []
        for c in np.unique(lbl):
            idx_c = np.where(lbl == c)[0]
            if self.shuffle: rng.shuffle(idx_c)
            per_class.append(np.array_split(idx_c, k_eff))

        for i in range(k_eff):
            val = np.concatenate([per_class[j][i] for j in range(len(per_class))]) if len(per_class) else np.array([], dtype=int)
            mask = np.ones(n, dtype=bool); mask[val] = False
            tr = np.where(mask)[0]
            yield tr, val

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_pred.ndim == 1 or y_pred.shape[1] == 1:
        return float(np.mean((y_pred.ravel() > 0.5).astype(int) == y_true.ravel().astype(int)))
    yhat = np.argmax(y_pred, axis=1)
    ytru = np.argmax(y_true, axis=1) if (y_true.ndim==2 and y_true.shape[1]>1) else y_true.ravel().astype(int)
    return float(np.mean(yhat == ytru))

def cross_validate(model_factory: Callable[[], MLP],
                   X: np.ndarray, Y: np.ndarray,
                   splitter, fit_kwargs: Optional[Dict[str,Any]] = None,
                   scoring: Callable[[np.ndarray,np.ndarray], float] = accuracy_score) -> Dict[str,Any]:
    fit_kwargs = fit_kwargs or {}
    scores = []
    for tr, va in splitter.split(X, Y):
        if va.size == 0 or tr.size == 0:  # <- salta folds vacíos
            continue
        model = model_factory()
        model.fit(X[tr], Y[tr], **fit_kwargs)
        y_pred = model.predict(X[va])
        scores.append(scoring(Y[va], y_pred))
    return {"scores": np.array(scores),
            "mean": float(np.mean(scores)) if scores else float("nan"),
            "std": float(np.std(scores)) if scores else float("nan")}
