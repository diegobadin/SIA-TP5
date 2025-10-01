import numpy as np

def step_function(x):
    """
        Step activation function. Maps the weighted sum to -1 or 1.
    """
    if x >= 0:
        return 1
    else:
        return -1

def linear_function(x):
    """
        Linear activation function. Identity functions.
    """
    return x

# --- Sigmoid family helpers for non-linear perceptron ---

def logistic_function_factory(beta: float = 1.0):
    """
    Returns the logistic activation θ(h) = 1 / (1 + exp(-2βh)) and its derivative
    θ'(h) = 2β θ(h) (1 - θ(h)).
    """
    def activation(h):
        return 1.0 / (1.0 + np.exp(-2.0 * beta * h))

    def derivative(h):
        y = activation(h)
        return 2.0 * beta * y * (1.0 - y)

    return activation, derivative


def tanh_function_factory(beta: float = 1.0):
    """
    Returns the hyperbolic tangent activation θ(h) = tanh(βh) and its derivative
    θ'(h) = β (1 - θ(h)^2).
    """
    def activation(h):
        return np.tanh(beta * h)

    def derivative(h):
        y = activation(h)
        return beta * (1.0 - y * y)

    return activation, derivative


def scaled_tanh_function_factory(beta: float, a: float, b: float):
    """
    Scaled tanh: maps h → y in [a,b] via y = a + (b-a)*(tanh(βh)+1)/2.
    Derivative: θ'(h) = (b-a)/2 * β * (1 - tanh(βh)^2).
    """
    def base(h):
        return np.tanh(beta * h)

    def activation(h):
        return a + (b - a) * (base(h) + 1.0) / 2.0

    def derivative(h):
        return (b - a) * 0.5 * beta * (1.0 - base(h) ** 2)

    return activation, derivative


def scaled_logistic_function_factory(beta: float, a: float, b: float):
    """
    Scaled logistic: maps h → y in [a,b] via y = a + (b-a)*σ(2βh), σ(x)=1/(1+e^{-x}).
    Derivative: θ'(h) = (b-a) * 2β * σ(2βh) * (1-σ(2βh)).
    """
    def sigma(h):
        return 1.0 / (1.0 + np.exp(-2.0 * beta * h))

    def activation(h):
        return a + (b - a) * sigma(h)

    def derivative(h):
        s = sigma(h)
        return (b - a) * 2.0 * beta * s * (1.0 - s)

    return activation, derivative