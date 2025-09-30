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