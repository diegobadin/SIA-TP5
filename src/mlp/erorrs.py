from src.mlp.activations import Activation
import numpy as np

class Loss:
    def value(self, y_hat: np.ndarray, y: np.ndarray) -> float: ...
    def delta_out(self, y_hat: np.ndarray, y: np.ndarray,
                  z_out: np.ndarray, act_out: Activation) -> np.ndarray: ...

class MSELoss(Loss):
    def value(self, y_hat, y) -> float:
        d = y_hat - y
        return 0.5 * float(np.mean(d * d))
    def delta_out(self, y_hat, y, z_out, act_out) -> np.ndarray:
        # (y_hat - y) ⊙ f'(z)
        return (y_hat - y) * act_out.df(z_out, y_hat)

class CrossEntropyLoss(Loss):
    def __init__(self, eps: float = 1e-12):
        self.eps = eps
    def value(self, y_hat, y) -> float:
        yh = np.clip(y_hat, self.eps, 1.0 - self.eps)
        if y.ndim == 1 or y.shape[1] == 1:  # binaria
            yb, yhb = y.ravel(), yh.ravel()
            return float(-np.mean(yb*np.log(yhb) + (1 - yb)*np.log(1 - yhb)))
        return float(-np.mean(np.sum(y * np.log(yh), axis=1)))  # one-hot
    def delta_out(self, y_hat, y, z_out, act_out) -> np.ndarray:
        # softmax+CE o sigmoid+BCE: y_hat - y
        return (y_hat - y)
