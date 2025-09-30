import os

from src.perceptron import Perceptron
from src.training_algorithms import linear_perceptron
from utils.activations import linear_function
from utils.parse_csv_data import parse_csv_data


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
        n_inputs=n_inputs, activation=linear_function, alpha=0.005,
        max_iter=100, training_algorithm=linear_perceptron
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



def run(csv_file_path: str, model_type: str):
    """Maps the model type to the corresponding runner function."""
    model_type = model_type.lower()
    if model_type == "lineal":
        run_linear_perceptron_regression(csv_file_path)
    elif model_type == "no_lineal":
        return
    else:
        print(f"Error: Model type '{model_type}' not recognized. Use 'lineal' or 'no_lineal'.")
