import numpy as np


def train_base(perceptron, x, y, update_rule_fn, convergence_criterion_fn,
               convergence_threshold=None, mode="online", batch_size=16, shuffle=True):
    """
    BASE FUNCTION: Implements the common structure of Stochastic Gradient Descent (SGD) training.

    Args:
        perceptron: The model instance.
        x, y: Data and labels.
        update_rule_fn: Function to inject the weight update logic (simple vs. linear).
        convergence_criterion_fn: Function to inject the final stopping condition logic.
        convergence_threshold: Specific threshold for the linear perceptron (SSE).
        mode: "online", "minibatch", or "batch".
        batch_size: Size of mini-batches (if mode is "minibatch").
        shuffle: Whether to shuffle data each epoch.
    """
    weights = list(perceptron.weights)
    alpha = perceptron.alpha
    max_iter = perceptron.max_iter

    best_weights = list(weights)
    min_error = float("inf")

    n_samples = len(x)
    n_features = len(weights)

    perceptron.error_history = []

    idx = np.arange(n_samples)

    for epoch in range(max_iter):
        if shuffle:
            np.random.shuffle(idx)

        epoch_error_acc = 0.0

        if mode == "batch":
            # 1) congelar pesos y acumular deltas para el dataset COMPLETO
            perceptron.weights = weights
            delta_w = np.zeros(n_features, dtype=float)
            for i in idx:
                y_pred = perceptron.predict(x[i])
                linear_error = y[i] - y_pred

                # alpha=1 y pesos temporales para medir "cuánto" cambiarían -- no deben cambiar aún en batch
                tmp_w = list(weights)
                metric_update, w_new = update_rule_fn(tmp_w, linear_error, x[i], 1.0, n_features)
                epoch_error_acc += metric_update

                # Acumular el delta propuesto
                for j in range(n_features):
                    delta_w[j] += (w_new[j] - tmp_w[j])

            # 2) aplicar UNA sola actualización al final de la época
            for j in range(n_features):
                weights[j] += perceptron.alpha * delta_w[j]



        elif mode == "minibatch":
            # 1) recorrer mini-batches y acumular deltas dentro de cada lote
            for start in range(0, n_samples, batch_size):
                batch = idx[start:start + batch_size]
                perceptron.weights = weights  # congelar durante el mini-lote
                delta_w = np.zeros(n_features, dtype=float)

                for i in batch:
                    y_pred = perceptron.predict(x[i])
                    linear_error = y[i] - y_pred

                    tmp_w = list(weights)
                    metric_update, w_new = update_rule_fn(tmp_w, linear_error, x[i], 1.0, n_features)
                    epoch_error_acc += metric_update
                    for j in range(n_features):
                        delta_w[j] += (w_new[j] - tmp_w[j])

                # 2) una sola actualización por mini-lote
                for j in range(n_features):
                    weights[j] += perceptron.alpha * delta_w[j]

        else:  # "online"
            for i in idx:
                perceptron.weights = weights
                y_pred = perceptron.predict(x[i])
                linear_error = y[i] - y_pred

                # INJECTED LOGIC: Calculates metric update and performs weight update
                metric_update, weights = update_rule_fn(weights, linear_error, x[i], alpha, n_features)
                epoch_error_acc += metric_update

        perceptron.error_history.append(epoch_error_acc)

        # Early Stopping: Track the best model based on the metric
        if epoch_error_acc < min_error:
            min_error = epoch_error_acc
            best_weights = list(weights)

        # INJECTED LOGIC: Check final stopping condition
        if convergence_criterion_fn(epoch_error_acc, convergence_threshold):
            perceptron.weights = best_weights
            perceptron.best_weights = best_weights
            perceptron.min_error = min_error
            return best_weights

    # Final return after max_iter reached
    perceptron.weights = best_weights
    perceptron.min_error = min_error
    return best_weights

