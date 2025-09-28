def simple_perceptron(perceptron, x, y):
    """
        Implements the Perceptron Learning Rule.

        :param perceptron: The Perceptron object containing weights, alpha, and max_iter.
        :param x: Input data matrix (samples x features, including the bias column).
        :param y: Target labels vector (expected output).
        :return: The final (best) weights found.
    """
    weights = perceptron.weights.copy()
    alpha = perceptron.alpha
    max_iter = perceptron.max_iter
    best_weights = weights.copy()
    min_error = float("inf")
    n_samples = len(x)
    n_features = len(weights)

    for epoch in range(max_iter):
        error_count = 0

        for i in range(n_samples):
            x_i = x[i]
            y_i = y[i]

            perceptron.weights = weights
            y_pred = perceptron.predict(x_i)

            error = y_i - y_pred

            if error != 0:
                error_count += 1
                for j in range(n_features):
                    weights[j] += alpha * error * x_i[j]

        if error_count < min_error:
            min_error = error_count
            best_weights = weights.copy()

        if error_count == 0:
            print(f"✅ Convergence reached at epoch {epoch + 1}.")
            perceptron.weights = best_weights
            perceptron.best_weights = best_weights
            perceptron.min_error = min_error
            return best_weights

        if epoch == max_iter - 1:
            print(f"⚠️ Max iterations of {max_iter} reached. Final error count: {error_count}")

    perceptron.weights = best_weights
    perceptron.best_weights = best_weights
    perceptron.min_error = min_error
    return best_weights