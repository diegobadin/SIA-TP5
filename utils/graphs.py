import os

import numpy as np
from matplotlib import pyplot as plt

from utils.metrics import accuracy_score, save_confusion_counts, save_confusion_with_heatmap
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
    plt.show(block=False)


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
    n_show_digits = min(n_show, len(X))
    n_rows = len(noise_levels)
    n_cols = n_show_digits * 2

    plt.figure(figsize=(12, n_rows * 1.5))  # Reducimos el alto de cada fila

    for i, noise_level in enumerate(noise_levels):
        X_noisy_level = X_noisy_all[i]
        for j in range(n_show_digits):
            # Con ruido
            plt.subplot(n_rows, n_cols, i * n_cols + j * 2 + 2)
            plt.imshow((X_noisy_level[j] >= 0.5).astype(int).reshape(shape), cmap='gray_r')
            plt.axis('off')

            # Solo para la primera fila, mostrar los números arriba
            if i == 0:
                plt.title(f"{j}", fontsize=10)

    plt.suptitle("Dígitos originales (izq) y con ruido (der) por nivel de ruido", fontsize=12)
    plt.subplots_adjust(hspace=0.1, wspace=0.05)  # Reducimos espacio entre filas y columnas
    plt.show(block=False)


def plot_accuracy_vs_noise(accuracies, noise_levels, fname="accuracy_vs_noise.png"):
    plt.figure(figsize=(7, 5))
    noise_percent = [n * 100 for n in noise_levels]

    plt.plot(noise_percent, accuracies, marker='o', linewidth=2)
    for x, y in zip(noise_percent, accuracies):
        plt.text(x, y + 0.01, f"{y:.2f}", ha='center', fontsize=9)

    plt.title("Precisión del modelo vs Nivel de ruido")
    plt.xlabel("Nivel de ruido (%)")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if SAVE_FIGS and fname:
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=140)

    plt.show(block=False)



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

        filepath = f"results/confusion_noise_{int(noise_level * 100)}.txt"
        class_labels = list(range(10))  # o nombres de tus dígitos si querés
        save_confusion_with_heatmap(
            y_true=Y,
            y_pred=predictions,
            filepath_prefix=f"results/confusion_noise_{int(noise_level * 100)}",
            comment=f"Nivel de ruido: {noise_level * 100:.0f}%",
            class_labels=class_labels
        )

        print(f"Ruido {int(noise_level * 100)}% → Precisión: {acc:.3f}")
        print(f"Resultados guardados en {filepath}\n")

    plot_digits_with_noise(X, X_noisy_all, noise_levels, n_show=n_show, shape=shape)
    plot_accuracy_vs_noise(accuracies, noise_levels)

    results = {
        "noise_levels": noise_levels,
        "accuracies": accuracies
    }
    return results

def plot_accuracy_folds(accuracies_dict, title="Accuracy por fold"):
    folds = np.arange(1, len(list(accuracies_dict.values())[0]) + 1)
    plt.figure(figsize=(8,5))
    for label, accs in accuracies_dict.items():
        plt.plot(folds, accs, marker='o', label=label)
    plt.xticks(folds)
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show(block=False)

def plot_accuracy_mean_std(accuracies_dict, title="Promedio de accuracy ± desviación"):
    labels = list(accuracies_dict.keys())
    means = [np.mean(accuracies_dict[l]) for l in labels]
    stds = [np.std(accuracies_dict[l]) for l in labels]

    plt.figure(figsize=(6,5))
    plt.bar(range(len(labels)), means, yerr=stds, capsize=5, color=['skyblue', 'lightgreen'])
    plt.xticks(range(len(labels)), labels)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(axis='y')
    plt.show()


def plot_mse_folds(mse_dict, title="MSE por fold"):
    folds = np.arange(1, len(list(mse_dict.values())[0]) + 1)
    plt.figure(figsize=(8,5))
    for label, mses in mse_dict.items():
        plt.plot(folds, mses, marker='o', label=label)
    plt.xticks(folds)
    plt.xlabel("Fold")
    plt.ylabel("MSE")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_mse_mean_std(mse_dict, title="Promedio ± desviación MSE"):
    labels = list(mse_dict.keys())
    means = [np.mean(mse_dict[l]) for l in labels]
    stds = [np.std(mse_dict[l]) for l in labels]

    plt.figure(figsize=(6,5))
    plt.bar(range(len(labels)), means, yerr=stds, capsize=5, color=['skyblue', 'lightgreen'])
    plt.xticks(range(len(labels)), labels)
    plt.ylabel("MSE")
    plt.title(title)
    plt.grid(axis='y')
    plt.show(block=False)

def plot_predictions_vs_real(y_true, y_pred_dict, title="Predicciones vs Reales"):
    plt.figure(figsize=(6,6))
    for label, y_pred in y_pred_dict.items():
        plt.scatter(y_true, y_pred, label=label)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', label='y=x')
    plt.xlabel("Valores reales")
    plt.ylabel("Predicciones")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

def plot_absolute_error(y_true, y_pred_dict, title="Error absoluto por muestra"):
    plt.figure(figsize=(8,5))
    x = np.arange(len(y_true))
    for label, y_pred in y_pred_dict.items():
        plt.plot(x, np.abs(y_true - y_pred), marker='o', label=label)
    plt.xlabel("Muestra")
    plt.ylabel("Error absoluto")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show(block=False)

def plot_decision_boundary_2(perceptron, x, y, title):
    x = np.array(x)
    y = np.array(y)
    weights = perceptron.weights

    # Frontera: w0 + w1*x1 + w2*x2 = 0  ->  x2 = -(w0 + w1*x1) / w2
    x1_vals = np.linspace(-1.5, 1.5, 100)
    x2_vals = -(weights[0] + weights[1]*x1_vals) / weights[2]

    plt.figure()
    plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], color='blue', label='1 (True)')
    plt.scatter(x[y == -1][:, 0], x[y == -1][:, 1], color='red', label='-1 (False)')
    plt.plot(x1_vals, x2_vals, color='green', label='Decision boundary')

    plt.title(f"Decision Boundary - {title}")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_xor_non_linear(x, y):
    plt.figure()
    x = np.array(x)
    y = np.array(y)

    plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], color='blue', label='1 (True)')
    plt.scatter(x[y == -1][:, 0], x[y == -1][:, 1], color='red', label='-1 (False)')

    plt.title("XOR - Not Linearly Separable")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.grid(True)
    plt.legend()
    plt.show()
