def train_base(perceptron, x, y, update_rule_fn, convergence_criterion_fn, convergence_threshold=None):
    """
    BASE FUNCTION: Implements the common structure of Stochastic Gradient Descent (SGD) training.

    Args:
        perceptron: The model instance.
        x, y: Data and labels.
        update_rule_fn: Function to inject the weight update logic (simple vs. linear).
        convergence_criterion_fn: Function to inject the final stopping condition logic.
        convergence_threshold: Specific threshold for the linear perceptron (SSE).
    """
    weights = list(perceptron.weights)
    alpha = perceptron.alpha
    max_iter = perceptron.max_iter

    best_weights = list(weights)
    min_error = float("inf")

    n_samples = len(x)
    n_features = len(weights)

    perceptron.error_history = []

    for epoch in range(max_iter):
        epoch_error_accumulator = 0

        for i in range(n_samples):
            x_i = x[i]
            y_i = y[i]

            perceptron.weights = weights
            y_pred = perceptron.predict(x_i)

            linear_error = y_i - y_pred

            # INJECTED LOGIC: Calculates metric update and performs weight update
            metric_update, weights = update_rule_fn(weights, linear_error, x_i, alpha, n_features)

            epoch_error_accumulator += metric_update

        # Early Stopping: Track the best model based on the metric
        if epoch_error_accumulator < min_error:
            min_error = epoch_error_accumulator
            best_weights = list(weights)

        # INJECTED LOGIC: Check final stopping condition
        if convergence_criterion_fn(epoch_error_accumulator, convergence_threshold):
            perceptron.weights = best_weights
            perceptron.best_weights = best_weights
            perceptron.min_error = min_error
            return best_weights

    # Final return after max_iter reached
    perceptron.weights = best_weights
    perceptron.min_error = min_error
    return best_weights

