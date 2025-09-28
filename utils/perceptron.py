import random

class Perceptron:
    def __init__(self, n_inputs, activation, alpha=0.1, max_iter=100, training_algorithm=None):
        self.n_inputs = n_inputs
        self.activation = activation
        self.alpha = alpha
        self.max_iter = max_iter
        self.training_algorithm = training_algorithm
        self.weights = [random.uniform(-0.05, 0.05) for _ in range(n_inputs + 1)]
        self.best_weights = list(self.weights)
        self.min_error = float("inf")

    def predict(self, x):
        """
            Calculates the weighted sum (excitation) manually, then applies activation.
            x must be a list of inputs (including the bias term '1' at index 0).
        """
        excitation = 0
        for w_i, x_i in zip(self.weights, x):
            excitation += w_i * x_i
        return self.activation(excitation)

    def fit(self, x, y):
        """
        Entrena el perceptrón con el algoritmo pasado como parámetro.
        """
        if self.training_algorithm is None:
            raise ValueError("Debe especificarse un algoritmo de entrenamiento.")
        return self.training_algorithm(self, x, y)
