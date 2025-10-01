import os

from src.perceptron import Perceptron
from src.training_algorithms import linear_perceptron, nonlinear_perceptron
from utils.activations import (
    linear_function,
    logistic_function_factory,
    tanh_function_factory,
    scaled_logistic_function_factory,
    scaled_tanh_function_factory,
)
from utils.parse_csv_data import parse_csv_data
import matplotlib.pyplot as plt


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



def run(csv_file_path: str, model_type: str):
    """Maps the model type to the corresponding runner function."""
    model_type = model_type.lower()
    if model_type == "lineal":
        run_linear_perceptron_regression(csv_file_path)
    elif model_type == "no_lineal":
        run_nonlinear_perceptron(csv_file_path)
    else:
        print(f"Error: Model type '{model_type}' not recognized. Use 'lineal' or 'no_lineal'.")


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
