import numpy as np
import matplotlib.pyplot as plt
from src.perceptron import Perceptron
from src.training_algorithms import linear_perceptron
from utils.activations import linear_function
from utils.parse_csv_data import parse_csv_data
from utils.splits import KFold


# === Exercise 2 ===

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



import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from src.perceptron import Perceptron
from src.training_algorithms import nonlinear_perceptron
from utils.activations import scaled_logistic_function_factory
from utils.parse_csv_data import parse_csv_data
from utils.metrics import cross_validate_regression, sse_score


import numpy as np
import matplotlib.pyplot as plt

from src.perceptron import Perceptron
from src.training_algorithms import nonlinear_perceptron
from utils.activations import scaled_logistic_function_factory
from utils.parse_csv_data import parse_csv_data
from utils.metrics import cross_validate_regression, sse_score
from utils.splits import KFold


def run_ex2_experimento4_cv_boxplot(
    csv_file_path: str,
    k: int = 5,
    eta: float = 0.01,
    epochs: int = 200,
    beta: float = 1.0,
    seed: int = 42,
):
    """
    🧪 Experimento 4 — Cross-validation (generalización)
    Modelo: perceptrón no lineal (sigmoide)
    Objetivo: evaluar capacidad de generalización → variabilidad del error de test entre folds.
    Gráfico: boxplot del MSE de test por fold.
    """

    X_list, Y_list = parse_csv_data(csv_file_path)
    X = np.array(X_list, dtype=float)
    Y = np.array(Y_list, dtype=float).ravel()

    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    n_inputs = Xb.shape[1] - 1  # sin contar el bias

    # salida en [0, 1]
    act, dact = scaled_logistic_function_factory(beta=beta, a=0.0, b=1.0)

    def nonlinear_factory():
        return Perceptron(
            n_inputs=n_inputs,
            activation=act,
            activation_derivative=dact,
            alpha=eta,
            max_iter=epochs,
            training_algorithm=nonlinear_perceptron,
            mode="online",
            shuffle=True,
        )

    splitter = KFold(n_splits=k, shuffle=True, seed=seed)

    sse_test_folds = cross_validate_regression(
        nonlinear_factory,
        Xb,
        Y,
        splitter,
        scoring=sse_score,
    )

    # Convertir SSE → MSE (aprox. n_test = N/k)
    n_test_approx = len(Y) / k
    mse_test_folds = np.array(sse_test_folds, dtype=float) / n_test_approx

    mean_mse = float(np.mean(mse_test_folds))
    std_mse = float(np.std(mse_test_folds))

    print("=== Experimento 4 — Cross-validation ===")
    print(f"Modelo: perceptrón no lineal (sigmoide)")
    print(f"k = {k}, η = {eta}, β = {beta}, épocas = {epochs}")
    print(f"MSE(test) medio = {mean_mse:.6f} ± {std_mse:.6f}")
    print(f"MSE(test) por fold: {', '.join(f'{v:.6f}' for v in mse_test_folds)}")

    plt.figure(figsize=(6, 5))
    plt.boxplot(mse_test_folds, vert=True, labels=[f"k={k}"])
    plt.ylabel("MSE (test)")
    plt.title("Cross-validation · MSE de test por fold (boxplot)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {
        "k": k,
        "eta": eta,
        "epochs": epochs,
        "beta": beta,
        "mse_test_folds": mse_test_folds.tolist(),
        "mean_mse": mean_mse,
        "std_mse": std_mse,
    }
