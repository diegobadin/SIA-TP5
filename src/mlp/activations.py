from __future__ import annotations
import numpy as np
from typing import Callable, List, Sequence, Tuple, Optional, Dict, Any, Iterator
class Activation:
    def __init__(self, f: Callable[[np.ndarray], np.ndarray],
                 df: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 name: str):
        self.f = f
        self.df = df
        self.name = name

def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
def _dsigmoid(z, a): return a * (1.0 - a)   # usar salida activada
def _tanh(z): return np.tanh(z)
def _dtanh(z, a): return 1.0 - a**2
def _relu(z): return np.maximum(0.0, z)
def _drelu(z, a): return (z > 0.0).astype(float)

def _softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)

SIGMOID = Activation(_sigmoid, _dsigmoid, "sigmoid")
TANH    = Activation(_tanh, _dtanh, "tanh")
RELU    = Activation(_relu, _drelu, "relu")
# Softmax: la derivada elemental no se usa; se maneja en la loss (CE)
SOFTMAX = Activation(_softmax, lambda z, a: np.ones_like(z), "softmax")
