from __future__ import annotations
import numpy as np
from typing import  List,Tuple

class Optimizer:
    def init(self, layer_shapes: List[Tuple[int,int]]) -> None: ...
    def begin_step(self) -> None: ...
    def update(self, layer_idx: int, grad: np.ndarray) -> np.ndarray: ...

class SGD(Optimizer):
    def __init__(self, lr: float = 1e-2): self.lr = lr
    def init(self, layer_shapes): pass
    def begin_step(self): pass
    def update(self, layer_idx, grad): return self.lr * grad

class Adam(Optimizer):
    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m: List[np.ndarray] = []; self.v: List[np.ndarray] = []; self.t = 0
    def init(self, layer_shapes): 
        self.m = [np.zeros(s) for s in layer_shapes]
        self.v = [np.zeros(s) for s in layer_shapes]
        self.t = 0
    def begin_step(self): self.t += 1  # un único t por update
    def update(self, layer_idx, grad):
        m = self.m[layer_idx] = self.b1*self.m[layer_idx] + (1-self.b1)*grad
        v = self.v[layer_idx] = self.b2*self.v[layer_idx] + (1-self.b2)*(grad*grad)
        m_hat = m / (1 - self.b1**self.t)
        v_hat = v / (1 - self.b2**self.t)
        return self.lr * (m_hat / (np.sqrt(v_hat) + self.eps))

class Momentum(Optimizer):
    def __init__(self, lr: float = 1e-2, beta: float = 0.9):
        self.lr = lr          # tasa de aprendizaje
        self.beta = beta      # coeficiente de momento (0..1)
        self.v: List[np.ndarray] = []  # “velocidad” por capa

    def init(self, layer_shapes: List[Tuple[int, int]]) -> None:
        # una matriz de ceros por capa con la misma shape que los pesos
        self.v = [np.zeros(s) for s in layer_shapes]

    def begin_step(self) -> None:
        # nada que hacer por update (a diferencia de Adam)
        pass

    def update(self, layer_idx: int, grad: np.ndarray) -> np.ndarray:
        # v_t = beta * v_{t-1} + grad
        self.v[layer_idx] = self.beta * self.v[layer_idx] + grad
        # paso = lr * v_t   (en tu MLP hacés: W -= paso)
        return self.lr * self.v[layer_idx]
