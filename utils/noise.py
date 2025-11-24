import numpy as np
from typing import Optional


def add_noise(X,
              noise_level=0.1,
              seed: Optional[int] = None,
              rng: Optional[np.random.Generator] = None,
              mode: str = "gaussian",
              salt_pepper_ratio: float = 0.05):
    """
    Agrega ruido a un conjunto de imágenes en [0,1] o [-1,1].

    Parámetros:
    - X: array de imágenes (shape: n_samples, n_features)
    - noise_level: desviación estándar del ruido gaussiano (si mode == 'gaussian')
    - seed / rng: controlan el generador pseudo-aleatorio
    - mode: 'gaussian' o 'salt_pepper'
    - salt_pepper_ratio: probabilidad de volcar un píxel a 0 o 1 (solo salt_pepper)
    """
    generator = rng if rng is not None else np.random.default_rng(seed)

    # Detectamos si los valores están en [-1,1] o [0,1]
    is_negative = X.min() < 0

    # Reescalamos a [0,1] si es necesario
    X_scaled = (X + 1) / 2 if is_negative else X.copy()

    if mode == "gaussian":
        noise = generator.normal(loc=0.0, scale=noise_level, size=X_scaled.shape)
        X_noisy = np.clip(X_scaled + noise, 0.0, 1.0)
    elif mode == "salt_pepper":
        X_noisy = X_scaled.copy()
        prob = generator.random(size=X_scaled.shape)
        half = salt_pepper_ratio / 2.0
        X_noisy[prob < half] = 0.0          # pepper
        X_noisy[(prob >= half) & (prob < salt_pepper_ratio)] = 1.0  # salt
    else:
        raise ValueError(f"Unknown noise mode '{mode}'. Use 'gaussian' or 'salt_pepper'.")

    if is_negative:
        X_noisy = X_noisy * 2 - 1

    return X_noisy
