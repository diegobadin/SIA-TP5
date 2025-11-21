import numpy as np
import matplotlib.pyplot as plt

from src.autoencoder import Autoencoder
from src.mlp.activations import SIGMOID, TANH
from src.mlp.erorrs import MSELoss
from src.mlp.optimizers import Adam
from utils.graphs import plot_loss
from utils.noise import add_noise
from utils.parse_font import parse_font_h


def _plot_grid(X, title, fname=None, shape=(7, 5), n_cols=8):
    n = X.shape[0]
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))
    plt.figure(figsize=(n_cols * 1.2, n_rows * 1.4))
    for i in range(n):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(X[i].reshape(shape), cmap="gray_r")
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    if fname:
        plt.savefig(f"outputs/{fname}", dpi=140)


def run(latent_dim=2, epochs=200, noise_level=0.0, deep=False,
        batch_size=4, lr=0.01, scale="-11"):
    """
    Entrena un autoencoder sobre los caracteres 7x5 de font.h.

    Parámetros frecuentes:
      latent_dim : tamaño del espacio latente
      noise_level: >0 para denoising (agrega ruido gaussiano a la entrada)
      deep       : agrega una capa oculta adicional 16-neuronas a cada lado
      scale      : '01' usa salida sigmoide, '-11' usa salida tanh
    """
    X, labels = parse_font_h(scale=scale)
    X_in = add_noise(X, noise_level=noise_level, seed=42) if noise_level > 0 else X

    hidden = [16] if deep else []
    out_act = SIGMOID if scale == "01" else TANH

    encoder_acts = [TANH] * (len(hidden) + 1)
    decoder_acts = ([TANH] * len(hidden)) + [out_act]

    autoencoder = Autoencoder(
        input_dim=X.shape[1],
        latent_dim=latent_dim,
        encoder_hidden=hidden,
        decoder_hidden=list(reversed(hidden)),
        encoder_activations=encoder_acts,
        decoder_activations=decoder_acts,
        loss=MSELoss(),
        optimizer=Adam(lr=lr),
        w_init_scale=0.1,
        seed=42,
    )

    losses = autoencoder.fit(
        X_in, X,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=True
    )

    # Reconstrucciones
    recon = autoencoder.predict(X)

    # Graficos básicos
    plot_loss(losses, loss_name="MSE", title="Autoencoder - Loss", fname="autoencoder_loss.png")
    _plot_grid(X, "Originales", fname="autoencoder_originals.png")
    _plot_grid(recon, "Reconstruidos", fname="autoencoder_recon.png")
    plt.show()

    # Reporte rápido
    final_loss = losses[-1] if losses else None
    print(f"[Autoencoder] latent={latent_dim} deep={deep} noise={noise_level} scale={scale}")
    print(f"Loss final: {final_loss:.6f}" if final_loss is not None else "Sin pérdidas registradas.")
