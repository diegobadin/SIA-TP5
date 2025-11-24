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
from utils.plot_autoencoder import plot_latent_space, plot_generation_results, plot_latent_space_with_generated, plot_grid


def encode_training_data(ae, X, labels):
    """
    Codifica todos los caracteres de entrenamiento y almacena pares (label, Z_i).
    
    Parámetros:
        ae: Autoencoder entrenado
        X: Datos de entrenamiento (n_samples, n_features)
        labels: Etiquetas de caracteres (n_samples,)
        
    Retorna:
        Diccionario que mapea etiquetas a vectores latentes
    """
    latent_vectors = ae.get_latent_representation(X)
    return {int(label): z for label, z in zip(labels, latent_vectors)}


def interpolate_chars(ae, X, idx1, idx2, alpha=0.5):
    """
    Interpola entre dos caracteres en el espacio latente.
    
    Parámetros:
        ae: Autoencoder entrenado
        X: Datos de entrenamiento (n_samples, n_features)
        idx1: Índice del primer carácter
        idx2: Índice del segundo carácter
        alpha: Factor de interpolación (0.0 = char2, 1.0 = char1, 0.5 = punto medio)
        
    Retorna:
        Tupla (z_new, X_new) donde z_new es el vector latente interpolado
        y X_new es el carácter decodificado
    """
    z1 = ae.encode(X[idx1])
    z2 = ae.encode(X[idx2])
    # Interpolación: alpha=0.0 -> char2, alpha=1.0 -> char1
    z_new = alpha * z1 + (1 - alpha) * z2
    X_new = ae.generate_from_latent(z_new)
    return z_new, X_new


def compare_with_training(X_new, X_train, labels, threshold=0.0):
    """
    Calcula la distancia de Hamming entre el carácter generado y todos los ejemplos de entrenamiento.
    
    Parámetros:
        X_new: Carácter generado (n_features,)
        X_train: Datos de entrenamiento (n_samples, n_features)
        labels: Etiquetas de entrenamiento (n_samples,)
        threshold: Umbral para binarización
        
    Retorna:
        Diccionario con distancias e información del vecino más cercano
    """
    # Binarizar
    X_new_bin = (X_new > threshold).astype(int)
    X_train_bin = (X_train > threshold).astype(int)
    
    # Calcular distancias de Hamming
    distances = np.sum(X_new_bin != X_train_bin, axis=1)
    
    # Encontrar vecino más cercano
    nearest_idx = np.argmin(distances)
    nearest_distance = distances[nearest_idx]
    nearest_label = int(labels[nearest_idx])
    
    # Crear diccionario de distancias
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
    Orquesta el flujo completo de generación.
    
    Parámetros:
        ae: Autoencoder entrenado
        X: Datos de entrenamiento
        labels: Etiquetas de caracteres
        char_idx1: Índice del primer carácter
        char_idx2: Índice del segundo carácter
        alpha: Factor de interpolación
        scale: Escala de datos ("-11" o "01")
        output_prefix: Prefijo para archivos de salida
        
    Retorna:
        Diccionario con resultados de generación
    """
    os.makedirs("outputs", exist_ok=True)
    
    # Paso 1: Codificar todos los caracteres de entrenamiento
    latent_dict = encode_training_data(ae, X, labels)
    all_latent = np.array([latent_dict[label] for label in sorted(latent_dict.keys())])
    
    # Paso 2: Interpolar entre dos caracteres
    z_new, X_new = interpolate_chars(ae, X, char_idx1, char_idx2, alpha)
    
    # Paso 3: Comparar con el conjunto de entrenamiento
    threshold = 0.0 if scale == "-11" else 0.5
    comparison = compare_with_training(X_new, X, labels, threshold)
    
    # Obtener el carácter vecino más cercano
    nearest_idx = comparison["nearest_neighbor"]["index"]
    nearest_char = X[nearest_idx]
    
    # Paso 4: Crear diccionario de mapeo de caracteres
    char_map = {
        0: '`', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g',
        8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o',
        16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w',
        24: 'x', 25: 'y', 26: 'z', 27: '{', 28: '|', 29: '}', 30: '~', 31: 'DEL'
    }
    
    # Paso 5: Graficar comparación (muestra el resultado visual)
    plot_generation_results(X_new, nearest_char, char_idx1, char_idx2, X,
                           f"Resultados de Generación (α={alpha})",
                           f"outputs/{output_prefix}_comparison.png")
    
    # Paso 6: Graficar espacio latente con punto generado resaltado (muestra dónde está en el espacio)
    plot_latent_space_with_generated(
        all_latent,
        np.array(sorted(latent_dict.keys())),
        z_new,
        f"Espacio Latente con Punto Generado (α={alpha}, {char_idx1}→{char_idx2})",
        f"outputs/{output_prefix}_latent_with_generated.png",
        char_map=char_map
    )
    
    # Guardar reporte de generación
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
    
    # Imprimir resumen
    print(f"\n=== Resultados de Generación ({output_prefix}) ===")
    print(f"Interpolado entre caracteres {char_idx1} y {char_idx2} (α={alpha})")
    print(f"Vecino más cercano: carácter {comparison['nearest_neighbor']['label']} "
          f"(distancia: {comparison['nearest_neighbor']['distance']} píxeles)")
    print(f"Distancia mínima: {comparison['min_distance']}, "
          f"Distancia máxima: {comparison['max_distance']}, "
          f"Distancia media: {comparison['mean_distance']:.2f}")
    print(f"Resultados guardados en outputs/{output_prefix}_*.png y *.json")
    
    return report


def run(latent_dim=2, noise_level=0.0):
    """
    Entrena un autoencoder sobre los caracteres 7x5 de font.h usando la configuración Arquitectura_Ancha_Rapida.
    
    Configuración fija: Arquitectura_Ancha_Rapida
    - Arquitectura: 35 → [30, 15] → 2 → [15, 30] → 35
    - Activaciones: TANH en encoder y capas ocultas, SIGMOID en salida
    - Optimizador: Adam(lr=0.001)
    - Batch size: 1
    - W init scale: 0.1
    - Scale: "01" (datos en [0, 1])
    - Early stopping: max_pixel_error=1.0

    Parámetros:
      latent_dim : tamaño del espacio latente (default: 2)
      noise_level: >0 para denoising (agrega ruido gaussiano a la entrada)
    """
    # Configuración Arquitectura_Ancha_Rapida
    scale = "01" 
    encoder_hidden = [30, 15]
    decoder_hidden = [15, 30]
    encoder_activations = [TANH, TANH, TANH]
    decoder_activations = [TANH, TANH, SIGMOID]
    lr = 0.001
    batch_size = 1
    w_init_scale = 0.1
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
    
    # Requisito 3: Graficar espacio latente (visualización 2D)
    if latent_dim == 2:
        latent = autoencoder.get_latent_representation(X)
        char_map = {
            0: '`', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g',
            8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o',
            16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w',
            24: 'x', 25: 'y', 26: 'z', 27: '{', 28: '|', 29: '}', 30: '~', 31: 'DEL'
        }
        plot_latent_space(
            {"Todos los Caracteres de Entrenamiento": latent},
            labels,
            save_path="outputs/latent_space.png",
            char_map=char_map,
            individual=True
        )
        print("\n[Requisito 3] Gráfico del espacio latente guardado en outputs/latent_space.png")
    
    # Requisito 4: Generar nuevas letras que no pertenecen al conjunto de entrenamiento
    if latent_dim == 2: 
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
        
        # Pares seleccionados basados en análisis del espacio latente de Arquitectura_Ancha_Rapida:
        # Estos pares fueron elegidos para demostrar diferentes aspectos de la generación:
        # Basados en la visualización del espacio latente donde se observa:
        # - Concentración de caracteres en la parte superior (z2 alto)
        # - Caracteres en la esquina inferior derecha (z1 alto, z2 bajo)
        # - Caracteres en la esquina inferior izquierda (z1 bajo, z2 bajo)
        char_pairs = [
            (16, 11),  # 'p' → 'k' - Transición suave entre letras similares (distancia ~0.15)
                      # Justificación: Ambos caracteres están en la parte superior del espacio latente,
                      # muy cercanos entre sí. Esta proximidad permite ver cómo el autoencoder genera
                      # variaciones sutiles entre caracteres vecinos, mostrando transiciones suaves
                      # y naturales en el espacio latente.
            
            (15, 13),  # 'o' → 'm' - Ambas con curvas (distancia ~0.53)
                      # Justificación: Ambas letras comparten características visuales (curvas redondas)
                      # pero están en diferentes regiones del espacio latente. La interpolación explora
                      # cómo se transforman estas características visuales mientras se mueve entre
                      # regiones, manteniendo cierta coherencia estructural.
            
            (1, 4),    # 'a' → 'd' - Esquinas opuestas (distancia ~1.06)
                      # Justificación: Caracteres diferentes ubicados en esquinas opuestas del espacio
                      # latente. 'a' está en la esquina inferior derecha y 'd' en la esquina inferior
                      # izquierda. Esta distancia permite explorar cómo el modelo generaliza entre
                      # regiones distantes, generando caracteres nuevos que no están en el conjunto
                      # de entrenamiento, manteniendo una transición visualmente coherente.
            
            (5, 21),   # 'e' → 'u' - Cruce vertical (distancia ~1.00)
                      # Justificación: Ambas letras tienen curvas pero están en regiones opuestas
                      # verticalmente del espacio latente. 'e' está en la parte superior-derecha
                      # y 'u' en la parte inferior-derecha. La interpolación cruza verticalmente
                      # el espacio, mostrando cómo el modelo transforma caracteres con características
                      # visuales similares (curvas) mientras se mueve entre regiones diferentes,
                      # generando transiciones interesantes y coherentes.
        ]
        
        # Obtener representaciones latentes para calcular distancias
        print("\nCalculando distancias en el espacio latente...")
        latent = autoencoder.get_latent_representation(X)
        
        # Calcular y mostrar distancias entre cada par
        print("\n" + "="*80)
        print("DISTANCIAS ENTRE PARES DE CARACTERES EN EL ESPACIO LATENTE")
        print("="*80)
        for idx, (idx1, idx2) in enumerate(char_pairs):
            if idx1 < len(X) and idx2 < len(X):
                char1 = char_map.get(int(labels[idx1]), str(int(labels[idx1])))
                char2 = char_map.get(int(labels[idx2]), str(int(labels[idx2])))
                
                # Calcular distancia euclidiana
                z1 = latent[idx1]
                z2 = latent[idx2]
                distance = np.linalg.norm(z1 - z2)
                
                print(f"\nPar {idx+1}: '{char1}' (idx {idx1}) ↔ '{char2}' (idx {idx2})")
                print(f"  Coordenadas '{char1}': z = [{z1[0]:.4f}, {z1[1]:.4f}]")
                print(f"  Coordenadas '{char2}': z = [{z2[0]:.4f}, {z2[1]:.4f}]")
                print(f"  Distancia euclidiana: {distance:.4f}")
        
        print("\n" + "="*80)
        print("INICIANDO GENERACIÓN DE CARACTERES")
        print("="*80)
        
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
    Ejecuta el ejercicio 1 con la configuración Arquitectura_Ancha_Rapida (fija).
    Esta configuración usa scale="01" (datos en [0, 1]) con SIGMOID en la salida.
    """
    print("\n" + "="*80)
    print("EJERCICIO 1: Autoencoder para caracteres 7x5")
    print("Configuración: Arquitectura_Ancha_Rapida (fija)")
    print("="*80 + "\n")
    
    # Ejecutar con la configuración Arquitectura_Ancha_Rapida (siempre se usa esta configuración)
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
