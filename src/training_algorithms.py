from utils.train_base import train_base

# --- RULE FOR SIMPLE PERCEPTRON  ---
def simple_perceptron_rule(weights, linear_error, x_i, alpha, n_features):
    """
    Updates weights only if a classification error occurs (error != 0).
    Returns 1 if there was an error (for error_count), 0 otherwise.
    """
    metric_update = 0
    if linear_error != 0:
        metric_update = 1  # The error metric is simply the count of misclassified samples
        # The update rule for the Simple Perceptron: W = W + alpha * Error * X
        for j in range(n_features):
            weights[j] += alpha * linear_error * x_i[j]
    return metric_update, weights

def simple_convergence_criterion(error_count):
    """Convergence criterion: Training stops when the error count is zero."""
    return error_count == 0


def simple_perceptron(perceptron, x, y, min_error_threshold=None, mode="online", batch_size=16, shuffle=True):
    """
    Trains the Perceptron for Classification using the Perceptron Learning Rule.
    """

    # Convergence condition for Simple Perceptron (error_count == 0)
    def convergence_criterion(error_count, threshold):
        return error_count == 0

    return train_base(
        perceptron, x, y,
        simple_perceptron_rule,
        convergence_criterion,
        convergence_threshold=min_error_threshold,  # Passed as None, but kept for signature consistency
        mode=mode,
        batch_size=batch_size,
        shuffle=shuffle
    )


# --- RULE FOR LINEAR PERCEPTRON (Delta Rule / Regression) ---
def linear_perceptron_rule(weights, linear_error, x_i, alpha, n_features):
    """
    Always updates weights (Gradient Descent) and returns the squared error (SSE) as the metric.
    """
    metric_update = linear_error ** 2  # The error metric is the Squared Error
    # The Delta Rule update: W = W + alpha * Error * X (always updates)
    for j in range(n_features):
        weights[j] += alpha * linear_error * x_i[j]
    return metric_update, weights

def linear_convergence_criterion(sse, min_error_threshold):
    """Convergence criterion: Training stops when the SSE falls below the defined threshold."""
    return sse < min_error_threshold


def linear_perceptron(perceptron, x, y, min_error_threshold,
                      mode="online", batch_size=16, shuffle=True):
    """
    Trains the Perceptron for Regression using the Delta Rule (Linear Perceptron).
    """

    # Convergence condition for Linear Perceptron (SSE < threshold)
    def convergence_criterion(sse, threshold):
        return sse < threshold

    return train_base(
        perceptron, x, y,
        linear_perceptron_rule,
        convergence_criterion,
        convergence_threshold=min_error_threshold,
        mode=mode,
        batch_size=batch_size,
        shuffle=shuffle

    )


def nonlinear_perceptron(perceptron, x, y, min_error_threshold, mode="online", batch_size=16, shuffle=True):
    """
    Trains the Perceptron with a sigmoid-like activation using gradient descent:
    Δw = α (y - o) θ'(h) x, minimizing SSE. Stops when SSE < threshold.
    """

    if perceptron.activation_derivative is None:
        raise ValueError("Se requiere 'activation_derivative' para el perceptrón no lineal.")

    activation_derivative = perceptron.activation_derivative

    def update_rule_with_derivative(weights, linear_error, x_i, alpha, n_features):
        # Compute excitation h = w · x
        h = 0
        for j in range(n_features):
            h += weights[j] * x_i[j]
        deriv = activation_derivative(h)
        metric_update = (linear_error ** 2)
        gradient_factor = alpha * linear_error * deriv
        for j in range(n_features):
            weights[j] += gradient_factor * x_i[j]
        return metric_update, weights

    def convergence_criterion(sse, threshold):
        return sse < threshold

    return train_base(
        perceptron, x, y,
        update_rule_with_derivative,
        convergence_criterion,
        convergence_threshold=min_error_threshold,
        mode=mode,
        batch_size=batch_size,
        shuffle=shuffle
    )