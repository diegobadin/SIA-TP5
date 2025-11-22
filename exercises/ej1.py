import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Agregar el directorio raíz del proyecto al path para poder importar módulos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.autoencoder import Autoencoder
from src.mlp.activations import SIGMOID, TANH
from src.mlp.erorrs import MSELoss
from src.mlp.optimizers import Adam
from utils.graphs import plot_loss
from utils.noise import add_noise
from utils.parse_font import parse_font_h
from utils.plot_ej1 import plot_grid
from utils.plot_autoencoder import plot_latent_space, plot_generation_results


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
    # Interpolación: alpha=0.0 -> char1, alpha=1.0 -> char2
    z_new = (1 - alpha) * z1 + alpha * z2
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




def demonstrate_generation(ae, X, labels, char_idx1, char_idx2, alpha=0.5, 
                          scale="01", output_prefix="generation"):
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
    # Crear diccionario con una sola configuración para plot_latent_space
    char_map = {
        0: '`', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g',
        8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o',
        16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w',
        24: 'x', 25: 'y', 26: 'z', 27: '{', 28: '|', 29: '}', 30: '~', 31: 'DEL'
    }
    plot_latent_space(
        {"Training Data": all_latent}, 
        np.array(sorted(latent_dict.keys())),
        save_path=f"outputs/{output_prefix}_latent_space.png",
        char_map=char_map,
        individual=True
    )
    
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
    # Agregar el punto generado al array de latent para visualizarlo
    all_latent_with_gen = np.vstack([all_latent, z_new])
    labels_with_gen = np.append(sorted(latent_dict.keys()), -1)  # -1 para el punto generado
    char_map_with_gen = char_map.copy()
    char_map_with_gen[-1] = "GEN"  # Etiqueta para el punto generado
    plot_latent_space(
        {f"Generated Point (α={alpha})": all_latent_with_gen},
        labels_with_gen,
        save_path=f"outputs/{output_prefix}_latent_with_generated.png",
        char_map=char_map_with_gen,
        individual=True
    )
    
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


def run(latent_dim=2, noise_level=0.0):
    """
    Entrena un autoencoder sobre los caracteres 7x5 de font.h usando la configuración Inicializacion_Grande.
    
    Configuración fija: Inicializacion_Grande
    - Arquitectura: 35 → [20, 10] → 2 → [10, 20] → 35
    - Activaciones: TANH en encoder y capas ocultas, SIGMOID en salida
    - Optimizador: Adam(lr=0.001)
    - Batch size: 1
    - W init scale: 0.2
    - Scale: "01" (datos en [0, 1])
    - Early stopping: max_pixel_error=1.0

    Parámetros:
      latent_dim : tamaño del espacio latente (default: 2)
      noise_level: >0 para denoising (agrega ruido gaussiano a la entrada)
    """
    # Configuración Inicializacion_Grande (fija)
    print("="*80)
    print("Configuración: Inicializacion_Grande")
    print("="*80)
    scale = "01" 
    encoder_hidden = [20, 10]
    decoder_hidden = [10, 20]
    encoder_activations = [TANH, TANH, TANH]
    decoder_activations = [TANH, TANH, SIGMOID]
    lr = 0.001
    batch_size = 1
    w_init_scale = 0.2
    max_epochs = 10000
    max_pixel_error = 1.0
    
    X, labels = parse_font_h(scale=scale)
    X_in = add_noise(X, noise_level=noise_level, seed=42) if noise_level > 0 else X

    autoencoder = Autoencoder(
        input_dim=X.shape[1],
        latent_dim=latent_dim,
        encoder_hidden=encoder_hidden,
        decoder_hidden=decoder_hidden,
        encoder_activations=encoder_activations,
        decoder_activations=decoder_activations,
        loss=MSELoss(),
        optimizer=Adam(lr=lr),
        w_init_scale=w_init_scale,
        seed=42,
    )

    losses = autoencoder.fit(
        X_in, X,
        epochs=max_epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=False,
        max_pixel_error=max_pixel_error,
        check_every=1,
        pixel_error_threshold=0.5 if scale == "01" else 0.0
    )

    # Reconstrucciones
    recon = autoencoder.predict(X)

    # Graficos básicos
    plot_loss(losses, loss_name="MSE", title="Autoencoder - Loss", fname="autoencoder_loss.png")
    plot_grid(X, "Originales", fname="autoencoder_originals.png")
    plot_grid(recon, "Reconstruidos", fname="autoencoder_recon.png")

    # Reporte rápido
    final_loss = losses[-1] if losses else None
    print(f"[Autoencoder] latent={latent_dim} noise={noise_level} scale={scale}")
    print(f"Loss final: {final_loss:.6f}" if final_loss is not None else "Sin pérdidas registradas.")
    
    # Requirement 3: Plot latent space (2D visualization)
    if latent_dim == 2:
        latent = autoencoder.get_latent_representation(X)
        char_map = {
            0: '`', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g',
            8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o',
            16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w',
            24: 'x', 25: 'y', 26: 'z', 27: '{', 28: '|', 29: '}', 30: '~', 31: 'DEL'
        }
        plot_latent_space(
            {"All Training Characters": latent},
            labels,
            save_path="outputs/latent_space.png",
            char_map=char_map,
            individual=True
        )
        print("\n[Requirement 3] Latent space plot saved to outputs/latent_space.png")
    
    # Requirement 4: Generate new letters not in training set
    if latent_dim == 2:  # Only generate if using 2D latent space
        print("\n" + "="*80)
        print("REQUIREMENT 4: Generar nuevas letras que no pertenecen al conjunto de entrenamiento")
        print("="*80)
        print("\nLa generación se realiza interpolando en el espacio latente entre dos caracteres")
        print("del conjunto de entrenamiento. Esto crea nuevos caracteres que no están en el dataset.")
        print("="*80)
        
        # Mapeo de índices a caracteres para mejor visualización
        char_map = {
            0: '`', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g',
            8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o',
            16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w',
            24: 'x', 25: 'y', 26: 'z', 27: '{', 28: '|', 29: '}', 30: '~', 31: 'DEL'
        }
        
        # Generate for several character pairs with different interpolation values
        char_pairs = [
            (1, 17),  # 'a' → 'q'
            (15, 13), # 'o' → 'm'
            (2, 3),   # 'b' → 'c'
            (2, 11),  # 'b' → 'k'
        ]
        
        # Diferentes valores de alpha para mostrar diferentes puntos de interpolación
        alphas = [0.3, 0.5, 0.7]
        
        for i, (idx1, idx2) in enumerate(char_pairs):
            if idx1 < len(X) and idx2 < len(X):
                char1 = char_map.get(int(labels[idx1]), str(int(labels[idx1])))
                char2 = char_map.get(int(labels[idx2]), str(int(labels[idx2])))
                print(f"\n--- Generando entre '{char1}' (índice {idx1}) y '{char2}' (índice {idx2}) ---")
                
                # Generar con diferentes valores de alpha
                for alpha in alphas:
                    demonstrate_generation(
                        autoencoder, X, labels, idx1, idx2, alpha=alpha,
                        scale=scale, output_prefix=f"gen_{i+1}_{char1}_to_{char2}_alpha{alpha}"
                    )
        
        print("\n" + "="*80)
        print("✓ Generación completada. Revisa los archivos en outputs/ para ver los resultados.")
        print("="*80)
    
    return autoencoder, X, labels, losses


if __name__ == "__main__":
    """
    Ejecuta el ejercicio 1 con la configuración Inicializacion_Grande (fija).
    Esta configuración usa scale="01" (datos en [0, 1]) con SIGMOID en la salida.
    """
    print("\n" + "="*80)
    print("EJERCICIO 1: Autoencoder para caracteres 7x5")
    print("Configuración: Inicializacion_Grande (fija)")
    print("="*80 + "\n")
    
    # Ejecutar con la configuración Inicializacion_Grande (siempre se usa esta configuración)
    autoencoder, X, labels, losses = run(
        latent_dim=2
    )
    
    print("\n" + "="*80)
    print("EJERCICIO COMPLETADO")
    print("="*80)
    print("\nArchivos generados en outputs/:")
    print("  - autoencoder_loss.png: Curva de pérdida durante el entrenamiento")
    print("  - autoencoder_originals.png: Caracteres originales")
    print("  - autoencoder_recon.png: Caracteres reconstruidos")
    print("  - latent_space.png: Visualización del espacio latente 2D")
    print("  - gen_*_*.png: Caracteres generados por interpolación")
    print("  - gen_*_generation_report.json: Reportes de generación")
    print("="*80)
