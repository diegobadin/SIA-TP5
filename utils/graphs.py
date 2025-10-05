import os

import numpy as np
from matplotlib import pyplot as plt

from utils.metrics import accuracy_score
from utils.noise import add_noise

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


def plot_kfold_accuracies(accs, title="K-Fold Accuracies"):
    import numpy as np, matplotlib.pyplot as plt
    xs = np.arange(1, len(accs)+1)
    mean, std = np.mean(accs), np.std(accs)

    plt.figure()
    plt.bar(xs, accs)
    plt.axhline(mean, linestyle="--", linewidth=2)
    plt.title(f"{title}\nmean={mean:.3f} ± {std:.3f}")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()


def plot_decision_boundary(model, X, Y, title):
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)          # (N,1) ⇒ reshape
    Z = (Z >= 0.5).astype(int)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.25, levels=[-0.5,0.5,1.5])
    plt.scatter(X[:,0], X[:,1], c=Y.ravel(), edgecolors="k", s=80)
    plt.title(title)
    plt.xlabel("x1"); plt.ylabel("x2"); plt.tight_layout(); plt.show()


def plot_digits_with_noise(X, X_noisy_all, noise_levels, n_show=10, shape=(7, 5)):
    plt.figure(figsize=(12, len(noise_levels) * 3))
    n_show_digits = min(n_show, len(X))

    for i, noise_level in enumerate(noise_levels):
        X_noisy_level = X_noisy_all[i]
        for j in range(n_show_digits):
            plt.subplot(len(noise_levels), n_show_digits * 2, i * n_show_digits * 2 + j * 2 + 1)
            plt.imshow((X[j] >= 0).astype(int).reshape(shape), cmap='gray_r')
            if i == 0:
                plt.title(f"{j}")
            plt.axis('off')

            # Con ruido
            plt.subplot(len(noise_levels), n_show_digits * 2, i * n_show_digits * 2 + j * 2 + 2)
            plt.imshow((X_noisy_level[j] >= 0.5).astype(int).reshape(shape), cmap='gray_r')
            plt.axis('off')

    plt.suptitle("Dígitos originales (izq) y con ruido (der) por nivel de ruido")
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    plt.show()


def plot_accuracy_vs_noise(accuracies, noise_levels):
    plt.figure()
    plt.plot([n * 100 for n in noise_levels], accuracies, marker='o')
    plt.title("Precisión del modelo vs Nivel de ruido")
    plt.xlabel("Nivel de ruido (%)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()


def evaluate_digits_with_noise(model, X, Y, noise_levels=None, n_show=10, shape=(7, 5)):
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.3, 0.5]

    accuracies = []
    X_bin = (X + 1) / 2  # desescalar a [0,1] para agregar ruido

    X_noisy_all = []
    for noise_level in noise_levels:
        X_noisy = add_noise(X_bin, noise_level=noise_level, seed=42)
        X_noisy_scaled = X_noisy * 2 - 1  # reescalar a [-1,1] para TANH

        predictions = model.predict(X_noisy_scaled)
        acc = accuracy_score(Y, predictions)
        accuracies.append(acc)
        X_noisy_all.append(X_noisy)

        print(f"Ruido {int(noise_level * 100)}% → Precisión: {acc:.3f}")

    plot_digits_with_noise(X, X_noisy_all, noise_levels, n_show=n_show, shape=shape)
    plot_accuracy_vs_noise(accuracies, noise_levels)
