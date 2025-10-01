# exercises/ex3.py
import numpy as np
import matplotlib.pyplot as plt
from src.mlp import MLP
from utils.activations import logistic_function_factory


def run():
    # Arquitectura: 2 entradas -> 3 ocultas -> 1 salida
    layer_sizes = [2, 3, 1]
    activations = [
        logistic_function_factory(),  # capa oculta
        logistic_function_factory()   # capa salida
    ]
    mlp = MLP(layer_sizes, activations, alpha=0.1, max_iter=5000)

    # Dataset XOR
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Y = [[0], [1], [1], [0]]

    # Historial de error por época
    mlp.error_history = []
    for epoch in range(mlp.max_iter):
        epoch_error = 0.0
        for x, y in zip(X, Y):
            mlp.backward(np.array(x), np.array(y))
            pred = mlp.predict(x)
            epoch_error += np.mean((np.array(y) - pred) ** 2)
        mlp.error_history.append(epoch_error / len(X))

    # Resultados finales
    print("\n=== XOR Results ===")
    correct = 0
    print(f"{'Input':<10} {'Expected':<10} {'Pred':<10} {'Class':<6}")
    print("-" * 40)
    for x, y in zip(X, Y):
        pred = mlp.predict(x)
        pred_class = int(pred[0] > 0.5)
        print(f"{x} {y[0]:<10} {pred[0]:<10.3f} {pred_class:<6}")
        if pred_class == y[0]:
            correct += 1

    accuracy = correct / len(Y)
    print(f"\nFinal Accuracy: {accuracy*100:.2f}%")

    # Graficar curva de error
    plt.plot(mlp.error_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training Error on XOR (MLP)")
    plt.show()
