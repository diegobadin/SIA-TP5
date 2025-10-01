import numpy as np


class Layer:
    def __init__(self, n_inputs, n_neurons, activation, activation_derivative):
        """
        n_inputs: cantidad de entradas (sin incluir bias).
        n_neurons: cantidad de neuronas en la capa.
        activation: función de activación para la capa.
        activation_derivative: derivada de la función de activación.
        """
        # +1 porque agregamos columna de bias
        self.weights = np.random.uniform(-0.05, 0.05, (n_neurons, n_inputs + 1))
        self.activation = activation
        self.activation_derivative = activation_derivative

        # Guardar valores de activación y pre-activación (para backprop)
        self.z = None  # pre-activación (W·x)
        self.a = None  # activación con bias agregado

    def forward(self, inputs):
        """
        inputs: vector de entradas (incluyendo bias).
        Devuelve la salida de la capa con bias agregado.
        """
        self.z = self.weights @ inputs  # producto matricial
        activated = np.array([self.activation(z_i) for z_i in self.z])
        # prepend bias = 1
        self.a = np.insert(activated, 0, 1.0)
        return self.a

    def compute_delta(self, delta_next, weights_next):
        """
        Backprop para capas ocultas.
        delta_next: vector de deltas de la capa siguiente (sin bias).
        weights_next: matriz de pesos de la capa siguiente (sin bias row).
        """
        deriv = np.array([self.activation_derivative(z_i) for z_i in self.z])
        # ignoramos la primera columna (bias) de los pesos de la siguiente capa
        return (weights_next[:, 1:].T @ delta_next) * deriv


class MLP:
    def __init__(self, layer_sizes, activations, alpha=0.1, max_iter=100):
        """
        layer_sizes: lista [n_inputs, n_hidden1, ..., n_outputs]
        activations: lista de tuplas (activation, derivative) para cada capa oculta y de salida
        alpha: learning rate
        """
        self.alpha = alpha
        self.max_iter = max_iter

        # Crear capas
        self.layers = []
        for i in range(1, len(layer_sizes)):
            n_inputs = layer_sizes[i - 1]
            n_neurons = layer_sizes[i]
            act, dact = activations[i - 1]
            self.layers.append(Layer(n_inputs, n_neurons, act, dact))

    def forward(self, x):
        """
        x: vector de entrada (sin bias).
        Devuelve la salida final de la red (sin bias).
        """
        # prepend bias = 1
        a = np.insert(x, 0, 1.0)
        for layer in self.layers:
            a = layer.forward(a)
        return a[1:]  # quitamos bias de la última capa

    def backward(self, x, y):
        """
        Backpropagation para un único patrón.
        x: entrada sin bias.
        y: salida esperada (vector).
        """
        # Paso forward para obtener activaciones
        self.forward(x)

        # Inicializar lista de deltas
        deltas = [None] * len(self.layers)

        # Delta en capa de salida
        last = self.layers[-1]
        error = y - last.a[1:]  # sin bias
        deriv = np.array([last.activation_derivative(z_i) for z_i in last.z])
        deltas[-1] = error * deriv

        # Deltas en capas ocultas
        for i in reversed(range(len(self.layers) - 1)):
            deltas[i] = self.layers[i].compute_delta(deltas[i + 1], self.layers[i + 1].weights)

        # Actualización de pesos
        a_prev = np.insert(x, 0, 1.0)  # entrada con bias
        for i, layer in enumerate(self.layers):
            if i > 0:
                a_prev = self.layers[i - 1].a
            # Regla: W = W + α * δ * a_prev^T
            layer.weights += self.alpha * np.outer(deltas[i], a_prev)

    def fit(self, X, Y):
        """
        X: lista de entradas (cada una sin bias).
        Y: lista de salidas esperadas.
        """
        for epoch in range(self.max_iter):
            for x, y in zip(X, Y):
                self.backward(np.array(x), np.array(y))

    def predict(self, x):
        return self.forward(x)
