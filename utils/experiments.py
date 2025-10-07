# ex2.py
import numpy as np
import matplotlib.pyplot as plt

from src.perceptron import Perceptron
from src.training_algorithms import linear_perceptron
from utils.activations import linear_function
from utils.parse_csv_data import parse_csv_data


def _add_bias(X: np.ndarray) -> np.ndarray:
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])


def _plot_mse(mse_history, lr, epochs):
    ep = np.arange(1, len(mse_history) + 1)
    plt.figure()
    plt.plot(ep, mse_history, linewidth=2)
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.title(f"Baseline lineal · MSE por época (η={lr}, épocas={epochs})")
    plt.tight_layout()
    plt.show()

def run_experimento1_baseline_lineal(csv_file_path: str,
                                     lr: float = 0.01,
                                     epochs: int = 200):
    """
        🧪 Experimento 1 — Baseline Lineal
        Modelo: perceptrón lineal (identidad)
        Entrenamiento: online
        η = 0.01 · Épocas = 200 · Init ∈ [-0.5, 0.5]
        Métrica: MSE por época
        """

    # === 1) Cargar datos usando tu parser ===
    X_list, Y_list = parse_csv_data(csv_file_path)
    if not X_list:
        print("No se pudieron cargar datos desde el CSV.")
        return

    X = np.array(X_list, dtype=float)
    Y = np.array(Y_list, dtype=float)
    Xb = _add_bias(X)

    # 2) perceptrón lineal (identidad), online
    model = Perceptron(
        n_inputs=Xb.shape[1] - 1,     # -1 por el bias explícito
        activation=linear_function,   # identidad
        alpha=lr,
        max_iter=epochs,
        training_algorithm=linear_perceptron,
        mode="online",
        shuffle=True
    )

    # 3) entrenar
    model.fit(Xb, Y)

    # 4) imprimir y graficar MSE
    print(f"[Experimento 1] Épocas: {epochs} | η={lr}")
    print(f"MSE inicial: {model.mse_history[0]:.6f}")
    print(f"MSE final:   {model.mse_history[-1]:.6f}")

    _plot_mse(model.mse_history, lr, epochs)

    return model