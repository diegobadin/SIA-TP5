from typing import Optional, Iterator, Tuple

import numpy as np


# ---------------------------
# Splits y métricas
# ---------------------------
# def holdout_split(X: np.ndarray, Y: np.ndarray, test_size: float = 0.2,
#                   shuffle: bool = True, seed: Optional[int] = None):
#     n = len(X); idx = np.arange(n)
#     if shuffle:
#         rng = np.random.default_rng(seed); rng.shuffle(idx)
#     cut = int(round(n*(1-test_size)))
#     tr, te = idx[:cut], idx[cut:]
#     return X[tr], X[te], Y[tr], Y[te]

import numpy as np

class HoldoutSplit:
    def __init__(self, test_ratio=0.2, shuffle=True, seed=None, stratify=False):
        self.test_ratio = float(test_ratio)
        self.shuffle = shuffle
        self.seed = seed
        self.stratify = stratify

    def __str__(self):
        return f"HoldoutSplit"

    __repr__ = __str__

    def split(self, X, y=None):
        """
        Rinde UN solo par (train_idx, test_idx), como un fold.
        - Si stratify=True, usa y (labels 1D) para mantener proporciones (si es posible).
        - Si no hay más de una muestra por clase, se puede desactivar stratify o ajustar test_ratio.
        """
        N = len(X)
        rng = np.random.default_rng(self.seed)

        if self.stratify:
            if y is None:
                raise ValueError("HoldoutSplit(stratify=True) requiere 'y' (labels 1D).")
            y = np.asarray(y).ravel()
            classes = np.unique(y)

            train_chunks, test_chunks = [], []
            for c in classes:
                idx_c = np.where(y == c)[0]
                if self.shuffle:
                    rng.shuffle(idx_c)
                n_c = len(idx_c)
                # al menos 1 test si hay >=2; evita dejar una clase sin train si n_c > 1
                n_test_c = int(round(n_c * self.test_ratio))
                n_test_c = min(max(n_test_c, 1 if n_c >= 2 else 0), n_c - 1 if n_c >= 2 else n_c)

                test_c = idx_c[:n_test_c]
                train_c = idx_c[n_test_c:]
                test_chunks.append(test_c)
                train_chunks.append(train_c)

            train_idx = np.concatenate(train_chunks) if train_chunks else np.array([], dtype=int)
            test_idx  = np.concatenate(test_chunks)  if test_chunks  else np.array([], dtype=int)
            if self.shuffle:
                rng.shuffle(train_idx)
                rng.shuffle(test_idx)
        else:
            idx = np.arange(N)
            if self.shuffle:
                rng.shuffle(idx)
            n_test = max(1, int(round(N * self.test_ratio)))
            n_test = min(n_test, N - 1) if N >= 2 else n_test
            test_idx  = idx[:n_test]
            train_idx = idx[n_test:]

        yield train_idx, test_idx


class KFold:
    def __init__(self, n_splits: int = 5, shuffle: bool = True, seed: Optional[int] = None):
        self.n_splits = n_splits
        assert n_splits >= 2
        self.k, self.shuffle, self.seed = n_splits, shuffle, seed

    def __str__(self):
        return f"KFold({self.n_splits})"

    __repr__ = __str__

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
        self.n_splits = n_splits
        assert n_splits >= 2
        self.k, self.shuffle, self.seed = n_splits, shuffle, seed

    def __str__(self):
        return f"StratifiedKFold({self.n_splits})"

    __repr__ = __str__
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
