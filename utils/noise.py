import numpy as np


def add_noise(X, noise_level=0.1, seed=None):
    """
    Agrega ruido gaussiano a un conjunto de imágenes en [0,1] o [-1,1].

    Parámetros:
    - X: array de imágenes (shape: n_samples, n_features)
    - noise_level: desviación estándar del ruido
    - seed: para reproducibilidad
    """
    if seed is not None:
        np.random.seed(seed)

    # Detectamos si los valores están en [-1,1] o [0,1]
    is_negative = X.min() < 0

    # Reescalamos a [0,1] si es necesario
    X_scaled = (X + 1) / 2 if is_negative else X.copy()

    # Ruido gaussiano
    noise = np.random.normal(loc=0.0, scale=noise_level, size=X_scaled.shape)
    X_noisy = X_scaled + noise

    # Clip para que siga en [0,1]
    X_noisy = np.clip(X_noisy, 0.0, 1.0)

    if is_negative:
        X_noisy = X_noisy * 2 - 1

    return X_noisy
