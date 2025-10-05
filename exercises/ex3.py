# exercises/ex3.py
import os
import numpy as np
import matplotlib.pyplot as plt

from src.mlp.activations import (
    TANH, SIGMOID, SOFTMAX
)

from src.mlp.erorrs import (
     CrossEntropyLoss, MSELoss
    
)
from src.mlp.mlp import MLP

from src.mlp.optimizers import (
    Adam, Momentum, SGD
)
from utils.graphs import plot_loss, plot_acc_curves, train_with_acc_curves, plot_kfold_accuracies, \
    plot_decision_boundary
from utils.metrics import cross_validate, accuracy_score

from utils.parse_csv_data import parse_digits_7x5_txt
from utils.splits import KFold, StratifiedKFold, HoldoutSplit


def run(item: str):
    item = (item or "").lower().strip()

    if item == "xor":
        # ---------- ITEM 1: XOR ----------
        X = np.array([[-1,  1],
                      [ 1, -1],
                      [-1, -1],
                      [ 1,  1]], dtype=float)
        Y = np.array([[1], [1], [0], [0]], dtype=float)

        model = MLP(
            layer_sizes=[2, 6, 1],
            activations=[TANH, SIGMOID],
            loss=MSELoss(),
            optimizer=Adam(lr=0.1),
            seed=7,
            w_init_scale=0.5
        )

        # Split 50/50 para poder graficar acc de train/test
        splitter = HoldoutSplit(test_ratio=0.5, shuffle=True, seed=7, stratify=True)

        tr_idx, te_idx = next(splitter.split(X, Y))

        # Construcción de sets con los índices; únicamente necesario en xor
        Xtr, Ytr = X[tr_idx], Y[tr_idx]
        Xte, Yte = X[te_idx], Y[te_idx]

        losses, tr_accs, te_accs = train_with_acc_curves(
            model, Xtr, Ytr, Xte, Yte,
            epochs=200, batch_size=1, shuffle=True, verbose=True
        )

        plot_loss(
            losses,
            loss_name=type(model.loss).__name__,
            title="XOR - Loss vs Época",
            fname="xor_loss.png",
        )
        plot_acc_curves(
            tr_accs, te_accs,
            title="XOR - Accuracy vs Época",
            fname="xor_acc.png",
        )
        plot_decision_boundary(model, X, Y, "XOR - Decision boundary")
        plt.show()

    elif item == "parity":
        # ---------- ITEM 2: PARITY ----------
        X, labels = parse_digits_7x5_txt("data/TP3-ej3-digitos_extra.txt")
        # Escalar a [-1,1] para TANH
        X = X * 2.0 - 1.0
        # 0 = par, 1 = impar
        Y = (labels % 2).astype(float).reshape(-1, 1)
        in_dim = X.shape[1]

        model = MLP(
            layer_sizes=[in_dim, 32, 1],
            activations=[TANH, SIGMOID],
            loss=CrossEntropyLoss(),      # BCE con sigmoid
            optimizer=Adam(lr=1e-3),
            seed=123,
            w_init_scale=0.2
        )

        # Split holdout estratificado 70/30 (rinde un único fold). Útil para graficar train/test.
        # splitter = HoldoutSplit(test_ratio=0.3, shuffle=True, seed=123, stratify=True)

        # splitter = KFold(n_splits=5, shuffle=True, seed=123)

        splitter = StratifiedKFold(n_splits=5, shuffle=True, seed=123)

        # --- Curvas con UN fold (para graficar train/test) ---
        tr_idx, te_idx = next(splitter.split(X, Y))
        Xtr, Ytr = X[tr_idx], Y[tr_idx]
        Xte, Yte = X[te_idx], Y[te_idx]

        losses, tr_accs, te_accs = train_with_acc_curves(
            model, Xtr, Ytr, Xte, Yte, epochs=50, batch_size=16, shuffle=True, verbose=True
        )
        plot_loss(losses,
                  loss_name=type(model.loss).__name__,
                  title="Parity - Loss vs Época",
                  fname="parity_loss.png")
        plot_acc_curves(tr_accs, te_accs, title="Parity - Accuracy vs Época", fname="parity_acc.png")

        # OJO: StratifiedKFold se puede usar nomás cuando tenemos más de una sola muestra de test por dígito
        # CV no tiene sentido si estamos corriendo con holdout
        res = cross_validate(
            model_factory=lambda: MLP([in_dim, 32, 1],[TANH, SIGMOID],
                                      CrossEntropyLoss(), Adam(lr=1e-3),
                                      seed=123, w_init_scale=0.2),
            X=X, Y=Y,
            splitter=splitter,
            fit_kwargs=dict(epochs=50, batch_size=16, verbose=False),
            scoring=accuracy_score,
        )
        title = f"Parity · {splitter}"
        plot_kfold_accuracies(res["folds"], title=title)
        plt.show()

    elif item == "digits":
        # ---------- ITEM 3: DIGITS ----------
        X, labels = parse_digits_7x5_txt("data/TP3-ej3-digitos_extra.txt")
        # Escalar a [-1,1] para TANH
        X = X * 2.0 - 1.0
        Y = np.eye(10, dtype=float)[labels]
        in_dim, out_dim = X.shape[1], Y.shape[1]

        model = MLP(
            layer_sizes=[in_dim, 32, 16, out_dim],
            activations=[TANH, TANH, SOFTMAX],
            loss=CrossEntropyLoss(),
            optimizer=Adam(lr=1e-3),
            seed=42,
            w_init_scale=0.2
        )

        # Split holdout estratificado 70/30 (rinde un único fold). Útil para graficar train/test.
        # splitter = HoldoutSplit(test_ratio=0.3, shuffle=True, seed=123, stratify=True)

        splitter = KFold(n_splits=5, shuffle=True, seed=123)

        # splitter = StratifiedKFold(n_splits=5, shuffle=True, seed=123)

        # --- Curvas con UN fold (para graficar train/test)
        tr_idx, te_idx = next(splitter.split(X, Y))
        Xtr, Ytr = X[tr_idx], Y[tr_idx]
        Xte, Yte = X[te_idx], Y[te_idx]

        losses, tr_accs, te_accs = train_with_acc_curves(
            model, Xtr, Ytr, Xte, Yte, epochs=50, batch_size=16, shuffle=True, verbose=True
        )
        plot_loss(losses, loss_name=type(model.loss).__name__, title="Digits - Loss vs Época", fname="digits_loss.png")
        plot_acc_curves(tr_accs, te_accs, title="Digits - Accuracy vs Época", fname="digits_acc.png")

        # OJO: StratifiedKFold se puede usar nomás cuando tenemos más de una sola muestra de test por dígito
        res = cross_validate(
            model_factory=lambda: MLP([in_dim, 32, 16, out_dim],
                                      [TANH, TANH, SOFTMAX],
                                      CrossEntropyLoss(), Adam(lr=1e-3),
                                      seed=42, w_init_scale=0.2),
            X=X, Y=Y,
            splitter=splitter,
            fit_kwargs=dict(epochs=50, batch_size=16, verbose=False),
            scoring=accuracy_score,
        )
        title = f"Digits · {splitter}"
        plot_kfold_accuracies(res["folds"], title=title)

        plt.show()

    else:
        print("Uso: python main.py ex3 <xor | parity | digits >")