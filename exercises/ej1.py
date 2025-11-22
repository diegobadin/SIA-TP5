import json
import os
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
        os.makedirs("outputs", exist_ok=True)
        plt.savefig(f"outputs/{fname}", dpi=140)


def _plot_latent(latent, labels, title, fname=None, highlight_point=None):
    """Plot latent space with character labels, optionally highlighting a generated point."""
    plt.figure(figsize=(8, 8))
    plt.scatter(latent[:, 0], latent[:, 1], c="C0", alpha=0.6, s=100, label="Training data")
    for i, (x, y) in enumerate(latent):
        plt.text(x, y, str(labels[i]), fontsize=8, ha="center", va="center")
    
    if highlight_point is not None:
        plt.scatter(highlight_point[0], highlight_point[1], c="red", s=200, 
                   marker="*", label="Generated", edgecolors="black", linewidths=2)
        plt.text(highlight_point[0], highlight_point[1], "GEN", fontsize=10, 
                ha="center", va="center", color="white", weight="bold")
    
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if fname:
        os.makedirs("outputs", exist_ok=True)
        plt.savefig(f"outputs/{fname}", dpi=140)


def encode_training_data(ae, X, labels):
    """
    Encode all training characters and store (label, Z_i) pairs.
    
    Args:
        ae: Trained autoencoder
        X: Training data (n_samples, n_features)
        labels: Character labels (n_samples,)
        
    Returns:
        Dictionary mapping labels to latent vectors
    """
    latent_vectors = ae.get_latent_representation(X)
    return {int(label): z for label, z in zip(labels, latent_vectors)}


def interpolate_chars(ae, X, idx1, idx2, alpha=0.5):
    """
    Interpolate between two characters in latent space.
    
    Args:
        ae: Trained autoencoder
        X: Training data (n_samples, n_features)
        idx1: Index of first character
        idx2: Index of second character
        alpha: Interpolation factor (0.0 = char1, 1.0 = char2, 0.5 = midpoint)
        
    Returns:
        Tuple (z_new, X_new) where z_new is the interpolated latent vector
        and X_new is the decoded character
    """
    z1 = ae.encode(X[idx1])
    z2 = ae.encode(X[idx2])
    z_new = alpha * z1 + (1 - alpha) * z2
    X_new = ae.generate_from_latent(z_new)
    return z_new, X_new


def compare_with_training(X_new, X_train, labels, threshold=0.0):
    """
    Compute Hamming distance between generated character and all training samples.
    
    Args:
        X_new: Generated character (n_features,)
        X_train: Training data (n_samples, n_features)
        labels: Training labels (n_samples,)
        threshold: Threshold for binarization
        
    Returns:
        Dictionary with distances and nearest neighbor info
    """
    # Binarize
    X_new_bin = (X_new > threshold).astype(int)
    X_train_bin = (X_train > threshold).astype(int)
    
    # Compute Hamming distances
    distances = np.sum(X_new_bin != X_train_bin, axis=1)
    
    # Find nearest neighbor
    nearest_idx = np.argmin(distances)
    nearest_distance = distances[nearest_idx]
    nearest_label = int(labels[nearest_idx])
    
    # Create distance dictionary
    distances_dict = {int(label): int(dist) for label, dist in zip(labels, distances)}
    
    return {
        "distances": distances_dict,
        "nearest_neighbor": {
            "index": int(nearest_idx),
            "label": nearest_label,
            "distance": int(nearest_distance)
        },
        "min_distance": int(nearest_distance),
        "max_distance": int(np.max(distances)),
        "mean_distance": float(np.mean(distances))
    }


def plot_generation_results(generated_char, nearest_neighbor_char, char_idx1, char_idx2, 
                           X_train, title, fname=None, shape=(7, 5)):
    """
    Plot generated character and nearest neighbor comparison.
    
    Args:
        generated_char: Generated character (n_features,)
        nearest_neighbor_char: Nearest neighbor from training (n_features,)
        char_idx1: Index of first character used in interpolation
        char_idx2: Index of second character used in interpolation
        X_train: Training data
        title: Plot title
        fname: Output filename
        shape: Shape to reshape characters for display
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original character 1
    axes[0].imshow(X_train[char_idx1].reshape(shape), cmap="gray_r")
    axes[0].set_title(f"Char {char_idx1}\n(original)")
    axes[0].axis("off")
    
    # Generated character
    axes[1].imshow(generated_char.reshape(shape), cmap="gray_r")
    axes[1].set_title(f"Generated\n(interp: {char_idx1}→{char_idx2})")
    axes[1].axis("off")
    
    # Original character 2
    axes[2].imshow(X_train[char_idx2].reshape(shape), cmap="gray_r")
    axes[2].set_title(f"Char {char_idx2}\n(original)")
    axes[2].axis("off")
    
    # Nearest neighbor
    axes[3].imshow(nearest_neighbor_char.reshape(shape), cmap="gray_r")
    axes[3].set_title("Nearest Neighbor\n(from training)")
    axes[3].axis("off")
    
    plt.suptitle(title)
    plt.tight_layout()
    if fname:
        os.makedirs("outputs", exist_ok=True)
        plt.savefig(f"outputs/{fname}", dpi=140)


def demonstrate_generation(ae, X, labels, char_idx1, char_idx2, alpha=0.5, 
                          scale="-11", output_prefix="generation"):
    """
    Orchestrate the complete generation workflow.
    
    Args:
        ae: Trained autoencoder
        X: Training data
        labels: Character labels
        char_idx1: Index of first character
        char_idx2: Index of second character
        alpha: Interpolation factor
        scale: Data scale ("-11" or "01")
        output_prefix: Prefix for output files
        
    Returns:
        Dictionary with generation results
    """
    os.makedirs("outputs", exist_ok=True)
    
    # Step 1: Encode all training characters
    latent_dict = encode_training_data(ae, X, labels)
    all_latent = np.array([latent_dict[label] for label in sorted(latent_dict.keys())])
    
    # Step 2: Plot latent space
    _plot_latent(all_latent, sorted(latent_dict.keys()), 
                "Latent Space - Training Data", 
                f"{output_prefix}_latent_space.png")
    
    # Step 3: Interpolate between two characters
    z_new, X_new = interpolate_chars(ae, X, char_idx1, char_idx2, alpha)
    
    # Step 4: Decode and save generated character
    plt.figure(figsize=(3, 4))
    plt.imshow(X_new.reshape(7, 5), cmap="gray_r")
    plt.title(f"Generated Character\n(α={alpha}, {char_idx1}→{char_idx2})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"outputs/{output_prefix}_generated_char.png", dpi=140)
    plt.close()
    
    # Step 5: Compare with training set
    threshold = 0.0 if scale == "-11" else 0.5
    comparison = compare_with_training(X_new, X, labels, threshold)
    
    # Get nearest neighbor character
    nearest_idx = comparison["nearest_neighbor"]["index"]
    nearest_char = X[nearest_idx]
    
    # Step 6: Display results
    plot_generation_results(X_new, nearest_char, char_idx1, char_idx2, X,
                           f"Generation Results (α={alpha})",
                           f"{output_prefix}_comparison.png")
    
    # Plot latent space with generated point highlighted
    _plot_latent(all_latent, sorted(latent_dict.keys()),
                f"Latent Space with Generated Point\n(α={alpha}, {char_idx1}→{char_idx2})",
                f"{output_prefix}_latent_with_generated.png",
                highlight_point=z_new)
    
    # Save generation report
    report = {
        "interpolation": {
            "char_idx1": int(char_idx1),
            "char_idx2": int(char_idx2),
            "alpha": float(alpha)
        },
        "generated_latent": z_new.tolist(),
        "comparison": comparison
    }
    
    with open(f"outputs/{output_prefix}_generation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"\n=== Generation Results ({output_prefix}) ===")
    print(f"Interpolated between characters {char_idx1} and {char_idx2} (α={alpha})")
    print(f"Nearest neighbor: character {comparison['nearest_neighbor']['label']} "
          f"(distance: {comparison['nearest_neighbor']['distance']} pixels)")
    print(f"Min distance: {comparison['min_distance']}, "
          f"Max distance: {comparison['max_distance']}, "
          f"Mean distance: {comparison['mean_distance']:.2f}")
    print(f"Results saved to outputs/{output_prefix}_*.png and *.json")
    
    return report


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

    # Reporte rápido
    final_loss = losses[-1] if losses else None
    print(f"[Autoencoder] latent={latent_dim} deep={deep} noise={noise_level} scale={scale}")
    print(f"Loss final: {final_loss:.6f}" if final_loss is not None else "Sin pérdidas registradas.")
    
    # Requirement 3: Plot latent space (2D visualization)
    if latent_dim == 2:
        latent = autoencoder.get_latent_representation(X)
        _plot_latent(latent, labels, "Latent Space - All Training Characters", 
                    "latent_space.png")
        print("\n[Requirement 3] Latent space plot saved to outputs/latent_space.png")
    
    # Requirement 4: Generate new letters not in training set
    if latent_dim == 2:  # Only generate if using 2D latent space
        print("\n=== Generating New Characters (Requirement 4) ===")
        
        # Generate for several character pairs
        char_pairs = [
            (1, 2),   # 'a' → 'b'
            (15, 4),  # 'o' → 'd'
            (2, 3),   # 'b' → 'c'
        ]
        
        for i, (idx1, idx2) in enumerate(char_pairs):
            if idx1 < len(X) and idx2 < len(X):
                demonstrate_generation(
                    autoencoder, X, labels, idx1, idx2, alpha=0.5,
                    scale=scale, output_prefix=f"gen_{i+1}_char{idx1}_to_char{idx2}"
                )
    
    plt.show()
