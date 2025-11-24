"""
VAE Training and Evaluation Script

This script implements the complete VAE workflow:
- Load/create emoji dataset
- Train VAE
- Visualize latent space
- Generate new samples
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

from src.vae import VAE, reparameterize, kl_divergence_loss
from src.mlp.activations import TANH, SIGMOID
from src.mlp.erorrs import MSELoss
from src.mlp.optimizers import Adam
from utils.parse_emoji import parse_emoji_txt
from utils.graphs import plot_loss


def plot_vae_training_curves(history, fname=None):
    """Plot VAE training curves (reconstruction, KL, total)."""
    epochs = range(1, len(history["total_loss"]) + 1)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["reconstruction_loss"], label="Reconstruction")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["kl_loss"], label="KL Divergence", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("KL Divergence Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["total_loss"], label="Total", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    if fname:
        os.makedirs("outputs", exist_ok=True)
        plt.savefig(f"outputs/{fname}", dpi=140)
    plt.close()


def visualize_latent_space(vae, X, labels, fname=None):
    """
    Visualize dataset in latent space using mean μ.
    
    Args:
        vae: Trained VAE
        X: Input data
        labels: Data labels
        fname: Output filename
    """
    mu = vae.get_latent_representation(X)
    
    if vae.latent_dim == 2:
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(mu[:, 0], mu[:, 1], c=labels, cmap="tab10", 
                             alpha=0.6, s=100, edgecolors="black", linewidths=0.5)
        plt.colorbar(scatter, label="Class")
        plt.xlabel("μ₁ (Latent Dimension 1)")
        plt.ylabel("μ₂ (Latent Dimension 2)")
        plt.title("VAE Latent Space (Mean μ)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if fname:
            os.makedirs("outputs", exist_ok=True)
            plt.savefig(f"outputs/{fname}", dpi=140)
        plt.close()
    else:
        print(f"Latent dimension is {vae.latent_dim}, cannot visualize directly.")


def plot_generated_samples(vae, X_train, n_generated=16, img_shape=(16, 16), fname=None):
    """
    Plot grid of generated samples and compare with training samples.
    
    Args:
        vae: Trained VAE
        X_train: Training samples
        n_generated: Number of samples to generate
        img_shape: Image shape (height, width)
        fname: Output filename
    """
    # Generate samples
    X_gen = vae.generate(n_generated)
    
    # Select representative training samples
    n_train_show = min(n_generated, len(X_train))
    train_indices = np.linspace(0, len(X_train)-1, n_train_show, dtype=int)
    X_train_show = X_train[train_indices]
    
    # Create comparison grid
    n_cols = 8
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4))
    
    # Top row: Training samples
    for i in range(n_cols):
        if i < len(X_train_show):
            axes[0, i].imshow(X_train_show[i].reshape(img_shape), cmap="gray_r")
        axes[0, i].set_title("Training" if i == 0 else "")
        axes[0, i].axis("off")
    
    # Bottom row: Generated samples
    for i in range(n_cols):
        if i < len(X_gen):
            axes[1, i].imshow(X_gen[i].reshape(img_shape), cmap="gray_r")
        axes[1, i].set_title("Generated" if i == 0 else "")
        axes[1, i].axis("off")
    
    plt.suptitle("Training Samples (top) vs Generated Samples (bottom)", fontsize=14)
    plt.tight_layout()
    if fname:
        os.makedirs("outputs", exist_ok=True)
        plt.savefig(f"outputs/{fname}", dpi=140)
    plt.close()


def plot_all_dataset_emojis(vae, X_train, labels, img_shape=(16, 16), fname=None):
    """
    Plot all emojis from the dataset with their VAE reconstructions.
    
    Args:
        vae: Trained VAE
        X_train: Training samples (all emojis)
        labels: Data labels
        img_shape: Image shape (height, width)
        fname: Output filename
    """
    # Get reconstructions for all training samples
    # forward() returns (x_recon, mu, log_var, z)
    X_recon, _, _, _ = vae.forward(X_train)
    
    # Determine grid layout: 8 columns × 4 rows (16 emojis total)
    # Row 1: Original emojis 0-7
    # Row 2: Original emojis 8-15
    # Row 3: Reconstructed emojis 0-7
    # Row 4: Reconstructed emojis 8-15
    n_emojis = len(X_train)
    n_cols = 8
    n_rows = 4
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))
    
    # Handle case where axes might be 1D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Row 1: Original emojis 0-7
    for i in range(n_cols):
        if i < n_emojis:
            axes[0, i].imshow(X_train[i].reshape(img_shape), cmap="gray_r")
            axes[0, i].set_title(f"Emoji {i}", fontsize=8)
        axes[0, i].axis("off")
    
    # Row 2: Original emojis 8-15
    for i in range(n_cols):
        emoji_idx = i + n_cols
        if emoji_idx < n_emojis:
            axes[1, i].imshow(X_train[emoji_idx].reshape(img_shape), cmap="gray_r")
            axes[1, i].set_title(f"Emoji {emoji_idx}", fontsize=8)
        axes[1, i].axis("off")
    
    # Row 3: Reconstructed emojis 0-7
    for i in range(n_cols):
        if i < n_emojis:
            axes[2, i].imshow(X_recon[i].reshape(img_shape), cmap="gray_r")
        axes[2, i].axis("off")
    
    # Row 4: Reconstructed emojis 8-15
    for i in range(n_cols):
        emoji_idx = i + n_cols
        if emoji_idx < n_emojis:
            axes[3, i].imshow(X_recon[emoji_idx].reshape(img_shape), cmap="gray_r")
        axes[3, i].axis("off")
    
    # Add row labels
    fig.text(0.02, 0.875, "Original", rotation=90, fontsize=12, va='center', ha='center')
    fig.text(0.02, 0.625, "Original", rotation=90, fontsize=12, va='center', ha='center')
    fig.text(0.02, 0.375, "Reconstructed", rotation=90, fontsize=12, va='center', ha='center')
    fig.text(0.02, 0.125, "Reconstructed", rotation=90, fontsize=12, va='center', ha='center')
    
    plt.suptitle("All Dataset Emojis: Original (rows 1-2) vs VAE Reconstruction (rows 3-4)", fontsize=14)
    plt.tight_layout(rect=[0.03, 0, 1, 1])  # Leave space for row labels
    if fname:
        os.makedirs("outputs", exist_ok=True)
        plt.savefig(f"outputs/{fname}", dpi=140)
    plt.close()


def train_vae_simple(X, latent_dim=2, epochs=200, beta=1.0, 
                    batch_size=4, lr=0.001, seed=42):
    """
    VAE training with proper backpropagation.
    
    This uses proper VAE training where:
    1. Reconstruction loss flows backward through decoder → ∂L_recon/∂z
    2. Gradients flow through reparameterization → ∂L_recon/∂μ, ∂L_recon/∂log_var
    3. KL loss directly affects encoder → ∂L_KL/∂μ, ∂L_KL/∂log_var
    4. Combined gradients backprop through encoder
    """
    input_dim = X.shape[1]
    
    # Create VAE
    vae = VAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        encoder_hidden=[32, 16],  # Hidden layers for encoder
        decoder_hidden=[16, 32],  # Hidden layers for decoder (symmetric)
        encoder_activations=[TANH, TANH],
        decoder_activations=[TANH, TANH, SIGMOID],  # TANH for hidden, SIGMOID for output
        reconstruction_loss=MSELoss(),
        optimizer=Adam(lr=lr),
        beta=beta,
        seed=seed
    )
    
    # Train using proper VAE backpropagation
    vae.fit(X, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=True)
    
    return vae


def run():
    """Main VAE training and evaluation workflow."""
    print("=" * 60)
    print("VARIATIONAL AUTOENCODER (VAE) TRAINING")
    print("=" * 60)
    
    # Load emoji dataset from ASCII art file
    print("\n1. Loading emoji dataset...")
    X, labels = parse_emoji_txt(path="data/emojis.txt", scale="01")
    print(f"   Dataset shape: {X.shape}")
    print(f"   Data range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"   Number of classes: {len(np.unique(labels))}")
    
    # Determine image shape from data
    img_size = int(np.sqrt(X.shape[1]))
    img_shape = (img_size, img_size)
    print(f"   Image shape: {img_shape}")
    
    # Train VAE
    print("\n2. Training VAE...")
    vae = train_vae_simple(
        X, 
        latent_dim=2,
        epochs=1000,
        beta=1,
        batch_size=4,
        lr=0.001,
        seed=42
    )
    
    # Plot training curves
    print("\n3. Plotting training curves...")
    plot_vae_training_curves(vae.history, "vae_training_curves.png")
    
    # Visualize latent space
    print("\n4. Visualizing latent space...")
    visualize_latent_space(vae, X, labels, "vae_latent_space.png")
    
    # Show all dataset emojis with reconstructions
    print("\n5. Plotting all dataset emojis with reconstructions...")
    plot_all_dataset_emojis(vae, X, labels, img_shape=img_shape, 
                            fname="vae_all_emojis_reconstruction.png")
    
    # Generate new samples
    print("\n6. Generating new samples...")
    plot_generated_samples(vae, X, n_generated=16, img_shape=img_shape, 
                          fname="vae_generated_samples.png")
    
    # Save report
    print("\n7. Saving results...")
    report = {
        "architecture": {
            "input_dim": vae.input_dim,
            "latent_dim": vae.latent_dim,
            "encoder_hidden": [32, 16],
            "decoder_hidden": [16, 32]
        },
        "training": {
            "epochs": 200,
            "beta": 0.1,
            "final_reconstruction_loss": vae.history["reconstruction_loss"][-1],
            "final_kl_loss": vae.history["kl_loss"][-1],
            "final_total_loss": vae.history["total_loss"][-1]
        }
    }
    
    with open("outputs/vae_training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "=" * 60)
    print("VAE TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final Reconstruction Loss: {vae.history['reconstruction_loss'][-1]:.6f}")
    print(f"Final KL Loss: {vae.history['kl_loss'][-1]:.6f}")
    print(f"Final Total Loss: {vae.history['total_loss'][-1]:.6f}")
    print("\nOutputs saved to outputs/vae_*.png and vae_training_report.json")


if __name__ == "__main__":
    run()

