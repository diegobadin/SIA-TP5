from src.perceptron import Perceptron
from src.training_algorithms import simple_perceptron
from utils.activations import step_function


def logical_gate(name, x_features, y_labels):
    """Initializes, trains, and tests the perceptron for a specific logic gate with clean output."""
    print(f"\n========================")
    print(f"  TESTING {name} GATE")
    print(f"========================")

    # 1. Add bias term
    x_with_bias = [[1] + row for row in x_features]

    # 2. Perceptron setup
    p = Perceptron(
        n_inputs=2,
        activation=step_function,
        alpha=0.1,
        max_iter=100,
        training_algorithm=simple_perceptron
    )

    # 3. Train
    best_weights = p.fit(x_with_bias, y_labels)
    formatted_weights = [f"{w:.2f}" for w in best_weights]

    print(f"Final Weights (B, w1, w2): {formatted_weights}")
    training_status = "Converged (Error = 0)" if p.min_error == 0 else f"Stopped (Errors = {p.min_error})"
    print(f"Training Status: {training_status}")

    # 4. Predictions
    correct_count = 0
    print("\n| Input (x1, x2) | Expected | Predicted | Status |")
    print("|----------------|----------|-----------|--------|")

    for i, x_full in enumerate(x_with_bias):
        input_features = x_full[1:]
        prediction = p.predict(x_full)
        is_correct = (prediction == y_labels[i])
        status = "✅ OK" if is_correct else "❌ FAIL"
        if is_correct:
            correct_count += 1

        input_str = f"[{input_features[0]}, {input_features[1]}]"
        print(f"| {input_str:<14} | {y_labels[i]:^8} | {prediction:^9} | {status:^6} |")

    print(f"\nFinal Accuracy: {correct_count}/{len(y_labels)}")


def run():
    x_base = [[-1, 1], [1, -1], [-1, -1], [1, 1]]

    # AND Gate
    y_and = [-1, -1, -1, 1]
    logical_gate("AND", x_base, y_and)

    # XOR Gate
    y_xor = [1, 1, -1, -1]
    logical_gate("XOR", x_base, y_xor)
