"""
Funciones de visualización para análisis de autoencoders.
Genera gráficos para presentación PPT mostrando:
- Curvas de pérdida por época
- Reconstrucciones de caracteres
- Visualización del espacio latente 2D
- Comparaciones entre configuraciones
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Any, Optional


def plot_loss_curves(histories: Dict[str, List[float]], save_path: str = "outputs/loss_curves.png",
                    title: str = "Evolución del Error de Píxeles Promedio por Época"):
    """Grafica las curvas de error de píxeles promedio para múltiples configuraciones."""
    plt.figure(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    for idx, (name, history) in enumerate(histories.items()):
        epochs = range(1, len(history) + 1)
        plt.plot(epochs, history, label=name, linewidth=2.5, alpha=0.8, color=colors[idx])
    
    plt.xlabel("Época", fontsize=13, fontweight='bold')
    plt.ylabel("Error de Píxeles Promedio", fontsize=13, fontweight='bold')
    plt.title(title, fontsize=15, fontweight='bold', pad=15)
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    # No usar escala logarítmica para error de píxeles (puede tener valores cercanos a 0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Gráfico guardado: {save_path}")
    plt.close()


def plot_reconstructions(X_original: np.ndarray, reconstructions: Dict[str, np.ndarray], 
                        labels: np.ndarray, n_samples: int = None, 
                        save_path: str = "outputs/reconstructions.png"):
    """Grafica reconstrucciones de caracteres para múltiples configuraciones."""
    # Si n_samples es None, mostrar todos los caracteres
    if n_samples is None:
        n_samples = len(X_original)
    
    n_configs = len(reconstructions)
    # Aumentar el tamaño de la figura para acomodar todos los caracteres
    fig_width = max(24, n_samples * 0.7)  # Ancho mínimo de 24, o 0.7 por carácter
    fig_height = (n_configs + 1) * 1.8  # Altura por fila
    
    # Crear figura con espacio extra a la izquierda para los labels
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(n_configs + 1, n_samples + 1, width_ratios=[1.5] + [1]*n_samples, 
                          height_ratios=[1]*(n_configs + 1), hspace=0.3, wspace=0.1)
    
    # Primera fila: originales
    # Label "Original" en la primera columna
    ax_label_0 = fig.add_subplot(gs[0, 0])
    ax_label_0.text(0.5, 0.5, "Original", fontsize=12, fontweight='bold', 
                    ha='center', va='center', rotation=0)
    ax_label_0.axis('off')
    
    # Imágenes originales
    for i in range(n_samples):
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(X_original[i].reshape(7, 5), cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f"Idx {i}", fontsize=8, fontweight='bold', pad=3)
        ax.axis('off')
    
    # Filas siguientes: reconstrucciones de cada configuración
    for row_idx, (config_name, X_recon) in enumerate(reconstructions.items(), start=1):
        # Label de configuración en la primera columna
        ax_label = fig.add_subplot(gs[row_idx, 0])
        config_label = config_name.replace('_', ' ')
        ax_label.text(0.5, 0.5, config_label, fontsize=11, fontweight='bold', 
                     ha='center', va='center', rotation=0, wrap=True)
        ax_label.axis('off')
        
        # Imágenes reconstruidas
        for i in range(n_samples):
            ax = fig.add_subplot(gs[row_idx, i + 1])
            ax.imshow(X_recon[i].reshape(7, 5), cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
            ax.set_title(f"{i}", fontsize=7, pad=2)
            ax.axis('off')
    
    plt.suptitle("Comparación de Reconstrucciones - Todos los Caracteres", 
                 fontsize=16, fontweight='bold', y=0.995)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Gráfico guardado: {save_path}")
    plt.close()


def plot_latent_space(latent_reps: Dict[str, np.ndarray], labels: np.ndarray,
                     save_path: str = "outputs/latent_space.png",
                     char_map: Optional[Dict[int, str]] = None,
                     individual: bool = False):
    """Grafica el espacio latente 2D para múltiples configuraciones.
    
    Args:
        latent_reps: Diccionario con nombre de configuración -> representaciones latentes
        labels: Array con los índices de los caracteres
        save_path: Ruta donde guardar el gráfico
        char_map: Diccionario opcional que mapea índice -> carácter (ej: {0: '`', 1: 'a', ...})
        individual: Si True, genera un gráfico individual por configuración. Si False, todos en uno.
    
    Normaliza el espacio latente al rango [0, 1] para todas las configuraciones,
    independientemente de la activación usada (TANH produce [-1,1], SIGMOID produce [0,1]).
    """
    if individual:
        # Generar gráficos individuales
        base_path = save_path.replace('.png', '')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        for config_name, latent in latent_reps.items():
            # Crear nombre de archivo individual
            safe_name = config_name.replace(' ', '_').replace('/', '_')
            individual_path = f"{base_path}_{safe_name}.png"
            
            _plot_single_latent_space(
                latent, labels, config_name, individual_path, char_map
            )
            print(f"✓ Gráfico individual guardado: {individual_path}")
        
        return
    
    # Gráfico combinado (comportamiento original)
    n_configs = len(latent_reps)
    n_cols = 3
    n_rows = (n_configs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_configs == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.flatten()
    
    for idx, (config_name, latent) in enumerate(latent_reps.items()):
        ax = axes[idx]
        _plot_latent_on_axis(ax, latent, labels, config_name, char_map)
    
    # Ocultar ejes extra
    for idx in range(n_configs, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle("Visualización del Espacio Latente 2D (Normalizado a [0, 1])", 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Gráfico guardado: {save_path}")
    plt.close()


def _plot_single_latent_space(latent: np.ndarray, labels: np.ndarray, 
                              config_name: str, save_path: str,
                              char_map: Optional[Dict[int, str]] = None):
    """Genera un gráfico individual del espacio latente para una configuración."""
    fig, ax = plt.subplots(figsize=(10, 10))
    _plot_latent_on_axis(ax, latent, labels, config_name, char_map)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def _plot_latent_on_axis(ax, latent: np.ndarray, labels: np.ndarray,
                         config_name: str, char_map: Optional[Dict[int, str]] = None):
    """Plotea el espacio latente en un eje dado con colores distintivos."""
    # Normalizar el espacio latente al rango [0, 1]
    latent_normalized = latent.copy()
    for dim in range(latent.shape[1]):
        dim_values = latent[:, dim]
        min_val = np.min(dim_values)
        max_val = np.max(dim_values)
        if max_val > min_val:
            latent_normalized[:, dim] = (dim_values - min_val) / (max_val - min_val)
        else:
            latent_normalized[:, dim] = 0.5
    
    # Usar colores distintos y vibrantes para cada carácter
    n_chars = len(latent_normalized)
    
    # Generar colores únicos para cada carácter sin repetición
    # Combinar múltiples colormaps para tener más colores distintos
    colors_list = []
    
    # Tab20 tiene 20 colores distintos
    colors_tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
    colors_list.extend(colors_tab20)
    
    # Set3 tiene 12 colores distintos
    colors_set3 = plt.cm.Set3(np.linspace(0, 1, 12))
    colors_list.extend(colors_set3)
    
    # Pastel1 tiene 9 colores
    colors_pastel1 = plt.cm.Pastel1(np.linspace(0, 1, 9))
    colors_list.extend(colors_pastel1)
    
    # Dark2 tiene 8 colores
    colors_dark2 = plt.cm.Dark2(np.linspace(0, 1, 8))
    colors_list.extend(colors_dark2)
    
    # Accent tiene 8 colores
    colors_accent = plt.cm.Accent(np.linspace(0, 1, 8))
    colors_list.extend(colors_accent)
    
    # Paired tiene 12 colores
    colors_paired = plt.cm.Paired(np.linspace(0, 1, 12))
    colors_list.extend(colors_paired)
    
    # Si aún necesitamos más colores, generar colores únicos usando HSV
    if n_chars > len(colors_list):
        n_extra = n_chars - len(colors_list)
        # Usar HSV para generar colores uniformemente distribuidos en el círculo cromático
        # Evitar valores muy saturados o muy oscuros para mantener buena visibilidad
        hues = np.linspace(0, 1, n_extra, endpoint=False)
        # Usar saturación y valor altos para colores vibrantes
        for hue in hues:
            # Convertir HSV a RGB
            import colorsys
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)  # Saturación 0.8, Valor 0.9
            colors_list.append((rgb[0], rgb[1], rgb[2], 1.0))
    
    # Asegurarse de que tenemos exactamente n_chars colores únicos
    # Si tenemos más de los necesarios, tomar solo los primeros n_chars
    colors_list = colors_list[:n_chars]
    
    # Scatter plot con colores distintos y vibrantes
    for i in range(n_chars):
        char_idx = int(labels[i])
        # Usar directamente colors_list[i] ya que garantizamos que tiene n_chars elementos
        color = colors_list[i]
        
        # Hacer los colores más saturados y vibrantes
        # Convertir a RGB y aumentar la saturación
        if len(color) == 4:  # RGBA
            r, g, b, a = color
            # Aumentar saturación (hacer más vibrante)
            max_rgb = max(r, g, b)
            if max_rgb > 0:
                r = min(1.0, r * 1.2)
                g = min(1.0, g * 1.2)
                b = min(1.0, b * 1.2)
            color = (r, g, b, a)
        
        ax.scatter(latent_normalized[i, 0], latent_normalized[i, 1], 
                  c=[color], s=250, alpha=0.85, edgecolors='black', linewidth=2.0,
                  zorder=3)
    
    # Anotar con los caracteres o índices
    for i, (x, y) in enumerate(latent_normalized):
        char_idx = int(labels[i])
        if char_map is not None and char_idx in char_map:
            label_text = char_map[char_idx]
        else:
            label_text = str(char_idx)
        
        # Usar el color del punto como fondo de la etiqueta (más claro)
        # Usar directamente colors_list[i] ya que garantizamos que tiene n_chars elementos
        color = colors_list[i]
        if len(color) == 4:
            r, g, b, a = color
            # Hacer el fondo más claro para mejor contraste
            bg_color = (min(1.0, r * 0.7 + 0.3), min(1.0, g * 0.7 + 0.3), min(1.0, b * 0.7 + 0.3), 0.9)
        else:
            bg_color = 'white'
        
        ax.text(x, y, label_text, fontsize=11, ha='center', va='center', 
               fontweight='bold', color='black', 
               bbox=dict(boxstyle='round,pad=0.5', facecolor=bg_color, 
                        edgecolor=color[:3] if len(color) >= 3 else 'black', 
                        linewidth=2.5, alpha=0.9),
               zorder=4)
    
    ax.set_xlabel("z1 (Dimensión Latente 1)", fontsize=13, fontweight='bold')
    ax.set_ylabel("z2 (Dimensión Latente 2)", fontsize=13, fontweight='bold')
    ax.set_title(config_name.replace('_', ' '), fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_aspect('equal', adjustable='box')
    
    # Agregar fondo ligeramente coloreado para mejor visualización
    ax.set_facecolor('#fafafa')


def plot_pixel_error_comparison_table(autoencoders: Dict[str, Any], X: np.ndarray, 
                                     labels: np.ndarray, 
                                     save_path: str = "outputs/pixel_error_comparison.png"):
    """Genera una tabla comparativa mostrando el error de píxeles por índice y configuración."""
    # Calcular errores de píxeles para cada configuración
    config_names = list(autoencoders.keys())
    pixel_errors_dict = {}
    
    # Determinar threshold basado en el rango de los datos
    # Si los datos están en [0, 1] (scale="01"), threshold=0.5
    # Si los datos están en [-1, 1] (scale="-11"), threshold=0.0
    data_min = float(np.min(X))
    threshold = 0.5 if data_min >= 0.0 else 0.0
    
    for config_name, autoencoder in autoencoders.items():
        pixel_errors = autoencoder.pixel_error(X, threshold=threshold)
        pixel_errors_dict[config_name] = pixel_errors
    
    # Preparar datos para la tabla
    n_configs = len(config_names)
    n_chars = len(labels)
    
    # Crear figura con tabla
    fig, ax = plt.subplots(figsize=(max(16, n_configs * 2.5), max(10, n_chars * 0.4 + 3)))
    ax.axis('tight')
    ax.axis('off')
    
    # Preparar datos de la tabla
    table_data = []
    
    # Encabezado
    header = ['Índice', 'Carácter'] + [name.replace('_', ' ') for name in config_names]
    table_data.append(header)
    
    # Filas de datos (por carácter)
    for idx in range(n_chars):
        row = [str(idx), str(labels[idx])]
        
        for config_name in config_names:
            error = int(pixel_errors_dict[config_name][idx])
            row.append(f"{error}")
        
        table_data.append(row)
    
    # Fila de totales (suma por configuración)
    totals_row = ['TOTAL', '']
    for config_name in config_names:
        total = int(np.sum(pixel_errors_dict[config_name]))
        totals_row.append(str(total))
    table_data.append(totals_row)
    
    # Fila de mean error (promedio por configuración)
    mean_row = ['MEAN', '']
    for config_name in config_names:
        mean = np.mean(pixel_errors_dict[config_name])
        mean_row.append(f"{mean:.2f}")
    table_data.append(mean_row)
    
    # Fila de max error (máximo por configuración)
    max_row = ['MAX', '']
    for config_name in config_names:
        max_err = int(np.max(pixel_errors_dict[config_name]))
        max_row.append(str(max_err))
    table_data.append(max_row)
    
    # Crear tabla
    col_widths = [0.06, 0.06] + [0.88 / n_configs] * n_configs
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(7 if n_configs > 4 else 8)
    table.scale(1, 1.3)
    
    # Colorear celdas según valores
    # Encabezado
    for j in range(len(header)):
        table[(0, j)].set_facecolor('#343a40')
        table[(0, j)].set_text_props(weight='bold', color='white', fontsize=9)
    
    # Filas de datos
    for i in range(1, n_chars + 1):
        # Índice y carácter
        table[(i, 0)].set_facecolor('#f8f9fa')
        table[(i, 1)].set_facecolor('#f8f9fa')
        table[(i, 0)].set_text_props(fontsize=8)
        table[(i, 1)].set_text_props(fontsize=8)
        
        # Errores por configuración
        for j, config_name in enumerate(config_names):
            col_idx = j + 2
            error = pixel_errors_dict[config_name][i - 1]
            if error <= 1:
                table[(i, col_idx)].set_facecolor('#d4edda')  # Verde claro
            else:
                table[(i, col_idx)].set_facecolor('#f8d7da')  # Rojo claro
            table[(i, col_idx)].set_text_props(fontsize=8, weight='bold' if error > 1 else 'normal')
    
    # Filas de resumen (TOTAL, MEAN, MAX)
    summary_rows = [n_chars + 1, n_chars + 2, n_chars + 3]
    for i in summary_rows:
        for j in range(len(header)):
            if j < 2:
                table[(i, j)].set_facecolor('#343a40')
                table[(i, j)].set_text_props(weight='bold', color='white', fontsize=9)
            else:
                table[(i, j)].set_facecolor('#ffc107')  # Amarillo para resúmenes
                table[(i, j)].set_text_props(weight='bold', fontsize=9)
    
    plt.title("Comparación de Error de Píxeles por Índice y Configuración", 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Gráfico guardado: {save_path}")
    plt.close()


def plot_comparison_table(results: Dict[str, Dict], save_path: str = "outputs/comparison_table.png"):
    """Crea una tabla visual comparando las configuraciones."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Preparar datos
    headers = ['Configuración', 'Épocas', 'Tiempo (s)', 'Loss Final', 'Max Error', 'Mean Error', 'Convergió']
    rows = []
    
    for name, data in results.items():
        converged_str = "✓ SÍ" if data['converged'] else "✗ NO"
        rows.append([
            name,
            f"{data['epochs']}",
            f"{data['time']:.2f}",
            f"{data['loss']:.6f}",
            f"{data['max_error']:.2f}",
            f"{data['mean_error']:.2f}",
            converged_str
        ])
    
    # Crear tabla
    table = ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center',
                    colWidths=[0.25, 0.1, 0.1, 0.12, 0.1, 0.1, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Colorear filas según convergencia
    for i in range(len(rows)):
        if results[list(results.keys())[i]]['converged']:
            for j in range(len(headers)):
                table[(i+1, j)].set_facecolor('#d4edda')  # Verde claro
        else:
            for j in range(len(headers)):
                table[(i+1, j)].set_facecolor('#f8d7da')  # Rojo claro
    
    # Estilo del encabezado
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#343a40')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    plt.title("Comparación de Configuraciones", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Gráfico guardado: {save_path}")
    plt.close()

