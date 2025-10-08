import os

import numpy as np

from src.perceptron import Perceptron
from src.training_algorithms import linear_perceptron, nonlinear_perceptron
from utils.activations import (
    linear_function,
    scaled_logistic_function_factory,
    scaled_tanh_function_factory, tanh_function_factory,
)
from utils.experiments import run_experimento1_baseline_lineal, run_ex2_experimento4_cv_boxplot
from utils.graphs import plot_accuracy_folds, plot_predictions_vs_real, plot_absolute_error
from utils.metrics import cross_validate_regression, sse_score
from utils.parse_csv_data import parse_csv_data
import matplotlib.pyplot as plt

from utils.splits import StratifiedKFold, KFold, HoldoutSplit


def run_linear_perceptron_regression(csv_file_path: str):
    """Runs the Delta Rule for Linear Regression."""

    # 1. DATA LOADING AND PREPARATION
    # Load feature (X) and target (Y) data from the specified CSV file.
    x_features, y_labels = parse_csv_data(csv_file_path)
    if not x_features: return

    # Prepend the bias term (1) to each feature vector (x0 = 1).
    x_with_bias = [[1] + row for row in x_features]
    # Determine the number of input features (excluding the bias).
    n_inputs = len(x_features[0])

    # 2. PERCEPTRON INITIALIZATION
    # Instantiate the Perceptron model with hyperparameters.
    p = Perceptron(
        n_inputs=n_inputs,
        activation=linear_function,
        alpha=0.005,
        max_iter=100,
        training_algorithm=linear_perceptron,
        mode="online", # If mode is online, batch_size is ignored
        batch_size=8,
        shuffle=True
    )

    # 3. CONSOLE OUTPUT (HEADER)
    print("==================================================")
    print("  LINEAR PERCEPTRON (REGRESSION - DELTA RULE)")
    print(f"  {len(x_features)} SAMPLES LOADED FROM: {os.path.basename(csv_file_path)}")
    print("==================================================")

    # 4. TRAINING
    # Execute the training algorithm (Delta Rule) to find optimal weights.
    p.fit(x_with_bias, y_labels)

    # 5. RESULTS SUMMARY
    # Display the final weights found by the training algorithm.
    formatted_weights = [f"{w:.3f}" for w in p.weights]
    print(f"Final Weights (B, w1, w2, w3): {formatted_weights}")

    # Display the final error metric (Sum of Squared Errors) achieved on the training set.
    print(f"Final SSE (Sum of Squared Errors): {p.min_error:.4f}")

    # 6. DETAILED PREDICTION TABLE
    print("\n| Input (x1, x2, x3) | Expected (y) | Predicted (ŷ) | Absolute Error |")
    print("|--------------------|--------------|---------------|----------------|")

    total_abs_error = 0
    # Iterate over the training samples to show individual predictions.
    for x_full, y_expected in zip(x_with_bias, y_labels):
        y_pred = p.predict(x_full)
        abs_error_val = abs(y_expected - y_pred)
        total_abs_error += abs_error_val
        input_features = x_full[1:]
        input_str = f"[{input_features[0]:.1f}, {input_features[1]:.1f}, {input_features[2]:.1f}]"
        print(f"| {input_str:<18} | {y_expected:^12.3f} | {y_pred:^13.3f} | {abs_error_val:^14.3f} |")

    # Graph the training error over epochs
    plt.plot(p.error_history)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title(f"Training Error ({p.mode})")
    plt.show()

def run_nonlinear_perceptron(csv_file_path: str, activation: str = "tanh", beta: float = 1.0):
    """
    Runs the non-linear perceptron (Delta Rule with sigmoid/tanh activation).
    activation: "tanh" maps to θ(h)=tanh(βh), "logistic" maps to θ(h)=1/(1+exp(-2βh)).
    """
    x_features, y_labels = parse_csv_data(csv_file_path)
    if not x_features: return

    # Standardize features to zero-mean / unit-variance per column (using utils.stats)
    from utils.stats import fmean, pstdev
    cols = list(zip(*x_features))
    means = [fmean(col) for col in cols]
    stdevs = [max(1e-8, pstdev(col)) for col in cols]
    x_std = [[(val - m)/s for val, m, s in zip(row, means, stdevs)] for row in x_features]

    x_with_bias = [[1] + row for row in x_std]
    n_inputs = len(x_std[0])

    # Build a scaled activation directly to map to [y_min, y_max]
    y_min = min(y_labels)
    y_max = max(y_labels)
    if activation == "tanh":
        act, dact = scaled_tanh_function_factory(beta, y_min, y_max)
        act_name = "tanh[scaled]"
    else:
        act, dact = scaled_logistic_function_factory(beta, y_min, y_max)
        act_name = "logistic[scaled]"

    # Adjust learning rate to compensate for scaled activation range
    base_alpha = 0.005
    range_span = max(1e-8, y_max - y_min)
    effective_alpha = base_alpha / range_span

    p = Perceptron(
        n_inputs=n_inputs,
        activation=act,
        activation_derivative=dact,
        alpha=effective_alpha,
        max_iter=500,
        training_algorithm=nonlinear_perceptron,
        mode="online",  # If mode is online, batch_size is ignored
        batch_size=8,
        shuffle=True
    )

    print("==================================================")
    print(f"  NON-LINEAR PERCEPTRON ({act_name.upper()} - β={beta})")
    print(f"  {len(x_features)} SAMPLES LOADED FROM: {os.path.basename(csv_file_path)}")
    print("==================================================")

    # Train using original y (activation already outputs in [y_min,y_max])
    p.fit(x_with_bias, y_labels)

    formatted_weights = [f"{w:.3f}" for w in p.weights]
    print(f"Final Weights (B, w1, w2, w3): {formatted_weights}")
    print(f"Final SSE (Sum of Squared Errors): {p.min_error:.4f}")

    print("\n| Input (x1, x2, x3) | Expected (y) | Predicted (ŷ) | Absolute Error |")
    print("|--------------------|--------------|---------------|----------------|")
    for x_full, y_expected in zip(x_with_bias, y_labels):
        y_pred = p.predict(x_full)
        abs_error_val = abs(y_expected - y_pred)
        input_features = x_full[1:]
        input_str = f"[{input_features[0]:.1f}, {input_features[1]:.1f}, {input_features[2]:.1f}]"
        print(f"| {input_str:<18} | {y_expected:^12.3f} | {y_pred:^13.3f} | {abs_error_val:^14.3f} |")

    # Graph the training error over epochs
    plt.plot(p.error_history)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title(f"Training Error ({p.mode})")
    plt.show()


def run_perceptron_experiment(csv_file_path: str, lr: float = 0.01, epochs: int = 100, beta: float = 1.0):

    # --- Cargar datos ---
    X, Y = parse_csv_data(csv_file_path)
    if not X:
        return
    X = np.array(X)
    Y = np.array(Y)

    # --- Agregar bias ---
    X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    n_inputs = X_bias.shape[1]

    # --- Definir splitter (solo KFold) ---
    splitter = KFold(n_splits=5, shuffle=True, seed=42)

    # --- Definir factories de perceptrones ---
    linear_factory = lambda: Perceptron(
        n_inputs=n_inputs,
        activation=linear_function,
        alpha=lr,
        max_iter=epochs,
        training_algorithm=linear_perceptron,
    )

    non_linear_factory = lambda: Perceptron(
        n_inputs=n_inputs,
        activation=lambda h: np.tanh(beta * h),
        activation_derivative=lambda h: 1 - np.tanh(beta * h) ** 2,
        alpha=lr,
        max_iter=epochs,
        training_algorithm=nonlinear_perceptron,
    )

    results = {}

    # --- Cross-validate (MSE por fold) ---
    mse_lin_folds = cross_validate_regression(linear_factory, X_bias, Y, splitter, scoring=sse_score)
    mse_nonlin_folds = cross_validate_regression(non_linear_factory, X_bias, Y, splitter, scoring=sse_score)

    results['Lineal'] = mse_lin_folds
    results['No Lineal'] = mse_nonlin_folds

    # --- Graficar MSE por fold ---
    plot_accuracy_folds(results, title="MSE por fold (KFold)")

    # --- Entrenar ambos modelos con todo el dataset para ver predicciones ---
    model_lin_full = linear_factory()
    model_lin_full.fit(X_bias, Y)
    y_pred_lin_full = np.array([model_lin_full.predict(x) for x in X_bias])

    model_nonlin_full = non_linear_factory()
    model_nonlin_full.fit(X_bias, Y)
    y_pred_nonlin_full = np.array([model_nonlin_full.predict(x) for x in X_bias])

    y_pred_dict_full = {'Lineal': y_pred_lin_full, 'No Lineal': y_pred_nonlin_full}

    # --- Graficar predicciones vs reales y errores absolutos ---
    plot_predictions_vs_real(Y, y_pred_dict_full, title="Predicciones vs Reales (Todo el dataset)")
    plot_absolute_error(Y, y_pred_dict_full, title="Error absoluto por muestra (Todo el dataset)")

    # --- Mostrar resumen de MSE ---
    for model_name, vals in results.items():
        print(f"{model_name}: mean MSE={np.mean(vals):.3f}, std MSE={np.std(vals):.3f}")

    return results


def run(csv_file_path: str, model_type: str):
    """Maps the model type to the corresponding runner function."""
    model_type = model_type.lower()
    if model_type == "lineal":
        run_linear_perceptron_regression(csv_file_path)
    elif model_type == "no_lineal":
        run_nonlinear_perceptron(csv_file_path)
    elif model_type == "experiment":
        run_perceptron_experiment(csv_file_path)
    elif model_type in ("baseline_lineal", "exp1"):
        run_experimento1_baseline_lineal(csv_file_path, lr=0.01, epochs=50)
    elif model_type in ("boxplot_cv", "exp4"):
        run_ex2_experimento4_cv_boxplot(csv_file_path,
                                        k=5, eta=0.01, epochs=200,
                                        beta=1.0,seed= 42)

    else:
        print(f"Error: Model type '{model_type}' not recognized. Use 'lineal' or 'no_lineal'.")

