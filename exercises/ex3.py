# exercises/ex3.py
import os
import numpy as np
import matplotlib.pyplot as plt

from src.mlp.activations import (
    TANH, SIGMOID, SOFTMAX
)
from src.mlp.mlp import (
    MLP,accuracy_score, cross_validate, StratifiedKFold, holdout_split
)
from src.mlp.erorrs import (
     CrossEntropyLoss, MSELoss
    
)

from src.mlp.optimizers import (
    Adam, Momentum, SGD
)

from utils.parse_csv_data import parse_digits_7x5_txt

# ===================== Config de gráficos =====================
SAVE_FIGS = True
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ===================== Helpers de gráficos =====================
def plot_loss(history, loss_name="Loss", title="Training loss", fname=None):
    if history is None or len(history) == 0:
        print("[plot_loss] No hay history para graficar.")
        return
    plt.figure()
    plt.plot(range(1, len(history) + 1), history)
    plt.xlabel("Época")
    plt.ylabel(loss_name)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if SAVE_FIGS and fname:
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=140)

def plot_cv_scores(res, title="CV scores", fname=None):
    scores = res.get("scores", None)
    if scores is None or len(scores) == 0:
        print("[plot_cv_scores] No hay scores para graficar.")
        return
    idx = np.arange(1, len(scores) + 1)
    plt.figure()
    plt.bar(idx, scores)
    plt.axhline(res["mean"], linestyle="--", linewidth=1)
    plt.xticks(idx, [f"Fold {i}" for i in idx])
    plt.ylabel("Accuracy")
    plt.title(f"{title} (mean={res['mean']:.3f} ± {res['std']:.3f})")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if SAVE_FIGS and fname:
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=140)

# ===================== Entrenamiento con curvas de ACC =====================
def train_with_acc_curves(model, Xtr, Ytr, Xte, Yte,
                          epochs=200, batch_size=16,
                          shuffle=True, verbose=False):
    """
    Entrena de a 1 época y mide accuracy en train y test por cada época.
    Devuelve: (losses, train_accs, test_accs).
    """
    losses, train_accs, test_accs = [], [], []
    for ep in range(epochs):
        # entreno UNA época
        last_loss = model.fit(Xtr, Ytr, epochs=1, batch_size=batch_size,
                              shuffle=shuffle, verbose=False)[-1]
        losses.append(last_loss)

        # métricas
        train_accs.append(accuracy_score(Ytr, model.predict(Xtr)))
        test_accs.append(accuracy_score(Yte, model.predict(Xte)))

        if verbose and ((ep + 1) % max(1, epochs // 10) == 0):
            print(f"[{ep+1:03d}/{epochs}] loss={last_loss:.4f} "
                  f"train_acc={train_accs[-1]:.3f} test_acc={test_accs[-1]:.3f}")
    return losses, train_accs, test_accs

def plot_acc_curves(train_accs, test_accs, title="Accuracy vs Época", fname=None):
    plt.figure()
    plt.plot(range(1, len(train_accs)+1), train_accs, label="Train")
    plt.plot(range(1, len(test_accs)+1),  test_accs,  label="Test")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if SAVE_FIGS and fname:
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=140)

# ===================== Runner de experimentos =====================
def run(item: str):
    item = (item or "").lower().strip()

    if item in ("xor"):
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
        Xtr, Xte, Ytr, Yte = holdout_split(X, Y, test_size=0.5, shuffle=True, seed=7)
        losses, tr_accs, te_accs = train_with_acc_curves(
            model, Xtr, Ytr, Xte, Yte, epochs=200, batch_size=1, shuffle=True, verbose=True
        )
        plot_loss(losses, loss_name=type(model.loss).__name__, title="XOR - Loss vs Época", fname="xor_loss.png")
        plot_acc_curves(tr_accs, te_accs, title="XOR - Accuracy vs Época", fname="xor_acc.png")

        plt.show()

    elif item in ("parity"):
        # ---------- ITEM 2: PARITY ----------
        X, labels = parse_digits_7x5_txt("data/TP3-ej3-digitos.txt")
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

        Xtr, Xte, Ytr, Yte = holdout_split(X, Y, test_size=0.3, shuffle=True, seed=123)
        losses, tr_accs, te_accs = train_with_acc_curves(
            model, Xtr, Ytr, Xte, Yte, epochs=250, batch_size=16, shuffle=True, verbose=True
        )
        plot_loss(losses, loss_name=type(model.loss).__name__, title="Parity - Loss vs Época", fname="parity_loss.png")
        plot_acc_curves(tr_accs, te_accs, title="Parity - Accuracy vs Época", fname="parity_acc.png")

        # (opcional) CV
        res = cross_validate(
            model_factory=lambda: MLP([in_dim, 32, 1],[TANH, SIGMOID],
                                      CrossEntropyLoss(), Adam(lr=1e-3),
                                      seed=123, w_init_scale=0.2),
            X=X, Y=Y,
            splitter=StratifiedKFold(n_splits=5, shuffle=True, seed=123),
            fit_kwargs=dict(epochs=250, batch_size=16, verbose=False)
        )
        print(f"[PARITY] CV mean acc: {res['mean']:.4f} ± {res['std']:.4f}")

        plt.show()

    elif item in ("digits"):
        # ---------- ITEM 3: DIGITS ----------
        X, labels = parse_digits_7x5_txt("data/TP3-ej3-digitos.txt")
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

        Xtr, Xte, Ytr, Yte = holdout_split(X, Y, test_size=0.3, shuffle=True, seed=42)
        losses, tr_accs, te_accs = train_with_acc_curves(
            model, Xtr, Ytr, Xte, Yte, epochs=300, batch_size=16, shuffle=True, verbose=True
        )
        plot_loss(losses, loss_name=type(model.loss).__name__, title="Digits - Loss vs Época", fname="digits_loss.png")
        plot_acc_curves(tr_accs, te_accs, title="Digits - Accuracy vs Época", fname="digits_acc.png")

        # (opcional) CV
        res = cross_validate(
            model_factory=lambda: MLP([in_dim, 32, 16, out_dim],
                                      [TANH, TANH, SOFTMAX],
                                      CrossEntropyLoss(), Adam(lr=1e-3),
                                      seed=42, w_init_scale=0.2),
            X=X, Y=Y,
            splitter=StratifiedKFold(n_splits=5, shuffle=True, seed=42),
            fit_kwargs=dict(epochs=300, batch_size=16, verbose=False)
        )
        print(f"[DIGITS] CV mean acc: {res['mean']:.4f} ± {res['std']:.4f}")

        plt.show()

    else:
        print("Uso: python main.py ex3 <xor | parity | digits >")