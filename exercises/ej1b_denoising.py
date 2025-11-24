import json
import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from src.autoencoder import Autoencoder
from src.mlp.activations import TANH, SIGMOID
from src.mlp.erorrs import MSELoss
from src.mlp.optimizers import Adam
from utils.graphs import plot_loss
from utils.noise import add_noise
from utils.parse_font import parse_font_h


def _binarize(X: np.ndarray, threshold: Optional[float]):
    """Devuelve X binarizado usando el threshold indicado."""
    if threshold is None:
        return X
    return np.where(X > threshold, 1.0, 0.0)


def evaluate_denoising(ae, X_clean, X_noisy, threshold=0.0):
    """
    Evaluate denoising performance of the autoencoder.
    
    Args:
        ae: Trained denoising autoencoder
        X_clean: Clean original data (n_samples, n_features)
        X_noisy: Noisy input data (n_samples, n_features)
        threshold: Threshold for binarization (0.0 for [-1,1], 0.5 for [0,1])
        
    Returns:
        Dictionary with metrics:
        - mse: Mean squared error between denoised and clean
        - pixel_error_mean: Mean pixel error (Hamming distance)
        - pixel_error_std: Standard deviation of pixel errors
        - pixel_errors: Per-sample pixel errors
        - mse_per_sample: Per-sample MSE
    """
    # Denoise the noisy inputs
    X_denoised = ae.predict(X_noisy)
    
    # Compute MSE
    mse = float(np.mean((X_clean - X_denoised) ** 2))
    mse_per_sample = np.mean((X_clean - X_denoised) ** 2, axis=1)
    
    # Compute pixel error (Hamming distance)
    X_clean_bin = (X_clean > threshold).astype(int)
    X_denoised_bin = (X_denoised > threshold).astype(int)
    pixel_errors = np.sum(X_clean_bin != X_denoised_bin, axis=1)
    
    return {
        "mse": mse,
        "mse_per_sample": mse_per_sample.tolist(),
        "pixel_error_mean": float(np.mean(pixel_errors)),
        "pixel_error_std": float(np.std(pixel_errors)),
        "pixel_errors": pixel_errors.tolist(),
        "pixel_error_min": int(np.min(pixel_errors)),
        "pixel_error_max": int(np.max(pixel_errors))
    }


def plot_denoising_comparison(X_original, X_noisy, X_denoised, noise_level, 
                              fname=None, shape=(7, 5), n_samples=8,
                              threshold: Optional[float] = None):
    """
    Plot side-by-side comparison: Original | Noisy | Denoised.
    
    Args:
        X_original: Clean original data
        X_noisy: Noisy input data
        X_denoised: Denoised output data
        noise_level: Noise level used (for title)
        fname: Output filename
        shape: Shape to reshape characters
        n_samples: Number of samples to show
    """
    n_samples = min(n_samples, len(X_original))
    fig, axes = plt.subplots(n_samples, 3, figsize=(9, n_samples * 1.5))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    X_orig_plot = _binarize(X_original, threshold)
    X_noisy_plot = X_noisy  # mantener escala de grises para visualizar el ruido
    X_denoised_plot = _binarize(X_denoised, threshold)

    for i in range(n_samples):
        # Original
        axes[i, 0].imshow(X_orig_plot[i].reshape(shape), cmap="gray_r")
        if i == 0:
            axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")
        
        # Noisy
        axes[i, 1].imshow(X_noisy_plot[i].reshape(shape), cmap="gray_r")
        if i == 0:
            axes[i, 1].set_title(f"Noisy (σ={noise_level:.2f})")
        axes[i, 1].axis("off")
        
        # Denoised
        axes[i, 2].imshow(X_denoised_plot[i].reshape(shape), cmap="gray_r")
        if i == 0:
            axes[i, 2].set_title("Denoised")
        axes[i, 2].axis("off")
    
    plt.suptitle(f"Denoising Comparison - Noise Level: {noise_level:.2f}", fontsize=14)
    plt.tight_layout()
    if fname:
        os.makedirs("outputs", exist_ok=True)
        plt.savefig(f"outputs/{fname}", dpi=140)
    plt.close()


def plot_denoising_performance(metrics_dict, fname=None):
    """
    Plot denoising performance curves: MSE and pixel error vs noise level.
    
    Args:
        metrics_dict: Dictionary mapping noise_level to metrics dict
        fname: Output filename
    """
    noise_levels = sorted(metrics_dict.keys())
    mse_values = [metrics_dict[nl]["mse"] for nl in noise_levels]
    pixel_error_means = [metrics_dict[nl]["pixel_error_mean"] for nl in noise_levels]
    pixel_error_stds = [metrics_dict[nl]["pixel_error_std"] for nl in noise_levels]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # MSE plot
    ax1.plot(noise_levels, mse_values, marker='o', linewidth=2, markersize=8)
    ax1.set_xlabel("Noise Level (σ)")
    ax1.set_ylabel("MSE")
    ax1.set_title("MSE vs Noise Level")
    ax1.grid(True, alpha=0.3)
    for nl, mse in zip(noise_levels, mse_values):
        ax1.text(nl, mse + 0.001, f"{mse:.4f}", ha='center', fontsize=9)
    
    # Pixel error plot
    ax2.errorbar(noise_levels, pixel_error_means, yerr=pixel_error_stds, 
                marker='o', linewidth=2, markersize=8, capsize=5)
    ax2.set_xlabel("Noise Level (σ)")
    ax2.set_ylabel("Mean Pixel Error")
    ax2.set_title("Pixel Error vs Noise Level")
    ax2.grid(True, alpha=0.3)
    for nl, pe in zip(noise_levels, pixel_error_means):
        ax2.text(nl, pe + 0.5, f"{pe:.2f}", ha='center', fontsize=9)
    
    plt.suptitle("Denoising Performance", fontsize=16)
    plt.tight_layout()
    if fname:
        os.makedirs("outputs", exist_ok=True)
        plt.savefig(f"outputs/{fname}", dpi=140)
    plt.close()


def plot_noise_level_grid(X_original, noise_levels, X_denoised_dict, 
                         char_indices=None, fname=None, shape=(7, 5),
                         threshold: Optional[float] = None):
    """
    Plot grid showing same characters at different noise levels.
    
    Args:
        X_original: Clean original data
        noise_levels: List of noise levels tested
        X_denoised_dict: Dictionary mapping noise_level to denoised outputs
        char_indices: Which characters to show (default: first 4)
        fname: Output filename
        shape: Shape to reshape characters
    """
    if char_indices is None:
        char_indices = list(range(min(4, len(X_original))))
    
    n_chars = len(char_indices)
    n_levels = len(noise_levels)
    
    fig, axes = plt.subplots(n_levels, n_chars * 3, 
                            figsize=(n_chars * 3 * 1.5, n_levels * 1.5))
    
    if n_levels == 1:
        axes = axes.reshape(1, -1)
    if n_chars == 1:
        axes = axes.reshape(-1, 1)
    
    X_original_plot = _binarize(X_original, threshold)

    for i, noise_level in enumerate(sorted(noise_levels)):
        X_denoised = X_denoised_dict[noise_level]
        X_noisy = add_noise(X_original, noise_level=noise_level, seed=42)
        X_noisy_plot = X_noisy  # mostrar el ruido con sus intensidades reales
        X_denoised_plot = _binarize(X_denoised, threshold)
        
        for j, char_idx in enumerate(char_indices):
            col_base = j * 3
            
            # Original
            axes[i, col_base].imshow(X_original_plot[char_idx].reshape(shape), cmap="gray_r")
            if i == 0:
                axes[i, col_base].set_title(f"Char {char_idx}\nOriginal")
            axes[i, col_base].axis("off")
            
            # Noisy
            axes[i, col_base + 1].imshow(X_noisy_plot[char_idx].reshape(shape), cmap="gray_r")
            if i == 0:
                axes[i, col_base + 1].set_title("Noisy")
            axes[i, col_base + 1].axis("off")
            if j == 0:
                axes[i, col_base + 1].set_ylabel(f"σ={noise_level:.2f}", fontsize=10)
            
            # Denoised
            axes[i, col_base + 2].imshow(X_denoised_plot[char_idx].reshape(shape), cmap="gray_r")
            if i == 0:
                axes[i, col_base + 2].set_title("Denoised")
            axes[i, col_base + 2].axis("off")
    
    plt.suptitle("Denoising at Different Noise Levels", fontsize=16)
    plt.tight_layout()
    if fname:
        os.makedirs("outputs", exist_ok=True)
        plt.savefig(f"outputs/{fname}", dpi=140)
    plt.close()


def study_denoising_autoencoder(noise_levels=None, latent_dim=2, epochs=300, 
                               deep=True, batch_size=4, lr=0.01, scale="-11",
                               seed=42):
    """
    Systematic study of denoising autoencoder across different noise levels.
    
    Architecture:
    - Encoder: [35] → [16] → [latent_dim] (if deep=True)
    - Decoder: [latent_dim] → [16] → [35] (if deep=True)
    - Activations: TANH for hidden layers, TANH/SIGMOID for output
    
    Args:
        noise_levels: List of noise levels to test (default: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        latent_dim: Latent space dimension
        epochs: Training epochs per noise level
        deep: Use hidden layer (16 neurons) if True, else direct encoding
        batch_size: Batch size for training
        lr: Learning rate
        scale: Data scale ("-11" for [-1,1] or "01" for [0,1])
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with complete study results
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    os.makedirs("outputs", exist_ok=True)
    
    # Load data
    X, labels = parse_font_h(scale=scale)
    threshold = 0.0 if scale == "-11" else 0.5
    
    print("=" * 60)
    print("DENOISING AUTOENCODER STUDY")
    print("=" * 60)
    print(f"Architecture: {'Deep' if deep else 'Basic'}")
    print(f"  Encoder: [35] → {'[16] → ' if deep else ''}[{latent_dim}]")
    print(f"  Decoder: [{latent_dim}] → {'[16] → ' if deep else ''}[35]")
    print(f"Noise levels: {noise_levels}")
    print(f"Epochs per level: {epochs}")
    print("=" * 60)
    
    results = {}
    X_denoised_dict = {}
    
    # Study each noise level
    for noise_level in noise_levels:
        print(f"\n--- Training at noise level σ={noise_level:.2f} ---")
        
        # Prepare noisy data
        X_noisy = add_noise(X, noise_level=noise_level, seed=seed)
        
        # Configure architecture
        hidden = [16] if deep else []
        out_act = SIGMOID if scale == "01" else TANH
        encoder_acts = [TANH] * (len(hidden) + 1)
        decoder_acts = ([TANH] * len(hidden)) + [out_act]
        
        # Create and train autoencoder
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
            seed=seed,
        )
        
        # Train with noisy input, clean target
        losses = autoencoder.fit(
            X_noisy, X,  # Noisy input, clean target
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=False
        )
        
        # Evaluate denoising performance
        metrics = evaluate_denoising(autoencoder, X, X_noisy, threshold)
        metrics["training_loss"] = losses
        metrics["final_loss"] = float(losses[-1]) if losses else None
        
        results[noise_level] = metrics
        X_denoised_dict[noise_level] = autoencoder.predict(X_noisy)
        
        # Plot comparison for this noise level
        plot_denoising_comparison(
            X, X_noisy, X_denoised_dict[noise_level], noise_level,
            fname=f"denoising_comparison_noise_{noise_level:.2f}.png",
            threshold=threshold
        )
        
        # Plot training loss
        plot_loss(
            losses, 
            loss_name="MSE", 
            title=f"Training Loss - Noise Level {noise_level:.2f}",
            fname=f"denoising_loss_noise_{noise_level:.2f}.png"
        )
        
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  Pixel error: {metrics['pixel_error_mean']:.2f} ± {metrics['pixel_error_std']:.2f}")
        print(f"  Final training loss: {metrics['final_loss']:.6f}")
    
    # Generate summary visualizations
    print("\n--- Generating summary visualizations ---")
    plot_denoising_performance(results, "denoising_performance_curves.png")
    plot_noise_level_grid(X, noise_levels, X_denoised_dict, 
                         fname="denoising_grid_all_levels.png",
                         threshold=threshold)
    
    # Save comprehensive report
    report = {
        "architecture": {
            "type": "deep" if deep else "basic",
            "encoder": f"[35] → {'[16] → ' if deep else ''}[{latent_dim}]",
            "decoder": f"[{latent_dim}] → {'[16] → ' if deep else ''}[35]",
            "activations": "TANH" + ("/SIGMOID" if scale == "01" else ""),
            "latent_dim": latent_dim
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "scale": scale
        },
        "noise_levels": noise_levels,
        "results": {str(nl): {
            "mse": results[nl]["mse"],
            "pixel_error_mean": results[nl]["pixel_error_mean"],
            "pixel_error_std": results[nl]["pixel_error_std"],
            "pixel_error_min": results[nl]["pixel_error_min"],
            "pixel_error_max": results[nl]["pixel_error_max"],
            "final_loss": results[nl]["final_loss"]
        } for nl in noise_levels}
    }
    
    with open("outputs/denoising_study_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("STUDY SUMMARY")
    print("=" * 60)
    print(f"{'Noise Level':<12} {'MSE':<12} {'Pixel Error':<15} {'Final Loss':<12}")
    print("-" * 60)
    for nl in sorted(noise_levels):
        r = results[nl]
        print(f"{nl:<12.2f} {r['mse']:<12.6f} {r['pixel_error_mean']:.2f}±{r['pixel_error_std']:.2f}     {r['final_loss']:<12.6f}")
    print("=" * 60)
    print("\nResults saved to outputs/denoising_*.png and denoising_study_report.json")
    
    return report


def run():
    """Main entry point for denoising study."""
    study_denoising_autoencoder(
        noise_levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        latent_dim=2,
        epochs=300,
        deep=True,
        batch_size=4,
        lr=0.01,
        scale="-11"
    )


if __name__ == "__main__":
    run()

