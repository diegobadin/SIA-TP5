import json
import os

import numpy as np
from matplotlib import pyplot as plt

from exercises.ej2_vae import plot_vae_training_curves, visualize_latent_space, plot_generated_samples
from src.mlp.activations import TANH, SIGMOID
from src.mlp.erorrs import MSELoss
from src.mlp.optimizers import Adam
from src.vae import VAE, kl_divergence_loss
from utils.parse_emoji import parse_emoji_txt

EXPERIMENT_CONFIGS = [
    {
        "name": "A_recon_strong",
        "description": "Casi AE clásico: prioriza reconstrucción",
        "latent_dim": 2,
        "beta": 0.0,
        "encoder_hidden": [64, 32],
        "decoder_hidden": [32, 64],
        "epochs": 1000,
        "batch_size": 4,
        "lr": 0.001,
        "seed": 42,
    },
    {
        "name": "B_tradeoff",
        "description": "Buen compromiso recon / estructura latente",
        "latent_dim": 2,
        "beta": 0.1,
        "encoder_hidden": [32, 16],
        "decoder_hidden": [16, 32],
        "epochs": 1000,
        "batch_size": 4,
        "lr": 0.001,
        "seed": 42,
    },
    {
        "name": "C_prior_strong",
        "description": "Prior fuerte: espacio latente más gaussiano",
        "latent_dim": 2,
        "beta": 1.0,
        "encoder_hidden": [32, 16],
        "decoder_hidden": [16, 32],
        "epochs": 1000,
        "batch_size": 4,
        "lr": 0.001,
        "seed": 42,
    }
]

def plot_reconstructions(vae, X, n_samples=8, img_shape=(16, 16), fname=None):
    """
    Muestra originales vs reconstrucciones para un subconjunto de X.
    Fila de arriba: originales, fila de abajo: reconstrucciones.
    """
    # Elegimos algunas muestras
    n_samples = min(n_samples, len(X))
    X_subset = X[:n_samples]

    X_recon, _, _, _ = vae.forward(X_subset)

    fig, axes = plt.subplots(2, n_samples, figsize=(1.5 * n_samples, 3))

    for i in range(n_samples):
        axes[0, i].imshow(X_subset[i].reshape(img_shape), cmap="gray_r")
        axes[0, i].set_title("Original" if i == 0 else "")
        axes[0, i].axis("off")

        axes[1, i].imshow(X_recon[i].reshape(img_shape), cmap="gray_r")
        axes[1, i].set_title("Reconstr." if i == 0 else "")
        axes[1, i].axis("off")

    plt.suptitle("Originales (arriba) vs Reconstrucciones (abajo)", fontsize=14)
    plt.tight_layout()
    if fname:
        os.makedirs("outputs", exist_ok=True)
        plt.savefig(f"outputs/{fname}", dpi=140)
    plt.close()

def train_vae_for_config(X, config):
    """
    Construye y entrena un VAE según la configuración dada.
    Devuelve el modelo entrenado.
    """
    input_dim = X.shape[1]

    latent_dim = config["latent_dim"]
    beta = config["beta"]
    encoder_hidden = config["encoder_hidden"]
    decoder_hidden = config["decoder_hidden"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    lr = config["lr"]
    seed = config.get("seed", 42)

    encoder_acts = [TANH] * len(encoder_hidden)
    decoder_acts = [TANH] * len(decoder_hidden) + [SIGMOID]

    vae = VAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        encoder_hidden=encoder_hidden,
        decoder_hidden=decoder_hidden,
        encoder_activations=encoder_acts,
        decoder_activations=decoder_acts,
        reconstruction_loss=MSELoss(),
        optimizer=Adam(lr=lr),
        beta=beta,
        seed=seed,
    )

    vae.fit(X, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=True)
    return vae


def run_experiments():
    """
    Ejecuta todas las configuraciones en EXPERIMENT_CONFIGS,
    guarda métricas y gráficos comparando las variantes del VAE.
    """
    print("=" * 60)
    print("EXPERIMENTOS VAE – DISTINTAS CONFIGURACIONES")
    print("=" * 60)

    # 1. Cargar dataset de emojis (igual que en run())
    print("\n1. Cargando dataset de emojis...")
    X, labels = parse_emoji_txt(path="data/emojis.txt", scale="01")
    print(f"   Dataset shape: {X.shape}")
    print(f"   Data range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"   Número de clases: {len(np.unique(labels))}")

    img_size = int(np.sqrt(X.shape[1]))
    img_shape = (img_size, img_size)

    os.makedirs("outputs", exist_ok=True)

    all_results = []

    # 2. Loop sobre configs
    for cfg in EXPERIMENT_CONFIGS:
        name = cfg["name"]
        print("\n" + "-" * 60)
        print(f"Entrenando configuración: {name}")
        print("-" * 60)
        print(f"Descripción: {cfg.get('description', '')}")

        vae = train_vae_for_config(X, cfg)

        # Métricas básicas de entrenamiento
        hist = vae.history
        final_recon = float(hist["reconstruction_loss"][-1])
        final_kl = float(hist["kl_loss"][-1])
        final_total = float(hist["total_loss"][-1])

        # Métrica de reconstrucción en todo el dataset
        X_recon, mu_full, log_var_full, _ = vae.forward(X)
        mse_full = float(np.mean((X - X_recon) ** 2))
        kl_full = kl_divergence_loss(mu_full, log_var_full)

        # Guardar gráficos específicos de esta config
        base_fname = f"exp_{name}"

        # 1) Curvas de entrenamiento
        plot_vae_training_curves(hist, fname=f"{base_fname}_training.png")

        # 2) Espacio latente (proyectado a 2 dims si es mayor)
        if vae.latent_dim >= 2:
            visualize_latent_space(
                vae, X, labels, fname=f"{base_fname}_latent.png"
            )

        # 3) Originales vs reconstrucciones
        plot_reconstructions(
            vae, X, n_samples=8, img_shape=img_shape,
            fname=f"{base_fname}_recon.png"
        )

        # 4) Muestras generadas desde N(0, I)
        plot_generated_samples(
            vae, X, n_generated=16, img_shape=img_shape,
            fname=f"{base_fname}_generated.png"
        )

        # Guardar resultados numéricos
        result = {
            "name": name,
            "description": cfg.get("description", ""),
            "latent_dim": cfg["latent_dim"],
            "beta": cfg["beta"],
            "encoder_hidden": cfg["encoder_hidden"],
            "decoder_hidden": cfg["decoder_hidden"],
            "epochs": cfg["epochs"],
            "batch_size": cfg["batch_size"],
            "lr": cfg["lr"],
            "final_reconstruction_loss_history": final_recon,
            "final_kl_loss_history": final_kl,
            "final_total_loss_history": final_total,
            "mse_reconstruction_full_dataset": mse_full,
            "kl_full_dataset": kl_full,
        }
        all_results.append(result)

    # 3. Guardar resumen global en JSON
    summary = {"experiments": all_results}
    with open("outputs/vae_experiments_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("EXPERIMENTOS VAE COMPLETOS")
    print("=" * 60)
    print("Resumen guardado en outputs/vae_experiments_summary.json")
    print("Gráficos guardados con prefijo outputs/exp_*")
