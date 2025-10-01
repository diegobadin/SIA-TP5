import numpy as np


class Perceptron:
    def __init__(self, n_inputs, activation, alpha=0.1, max_iter=100, training_algorithm=None,
                 activation_derivative=None, mode="online", batch_size=16, shuffle=True):
        self.n_inputs = n_inputs
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.alpha = alpha
        self.max_iter = max_iter
        self.training_algorithm = training_algorithm
        self.weights = np.random.uniform(-0.05, 0.05, n_inputs + 1)
        self.best_weights = self.weights.copy()
        self.min_error = float("inf")
        self.error_threshold = 0.0001
        self.mode = mode
        self.batch_size = batch_size
        self.shuffle = shuffle

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
        return self.training_algorithm(self, x, y, self.error_threshold,
                                       mode=self.mode, batch_size=self.batch_size, shuffle=self.shuffle)
