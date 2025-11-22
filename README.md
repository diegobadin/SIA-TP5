# SIA TP5 - Deep Learning

Este trabajo práctico tiene como objetivo implementar algoritmos de Autoencoders (Autoencoder Básico y 
Denoising Autoencoder) y Variational Autoencoder (VAE). El foco está en la representación de datos en un espacio 
latente de baja dimensión y la generación de nuevas muestras. 

## Prerrequisitos
- [Python](https://www.python.org/downloads/) instalado en el sistema.
- `pip` disponible en la terminal (`pip --version` para verificar).
- Archivos de datos necesarios, como font.h (para los caracteres $5 \times 7$)

## Construcción

Para construir el proyecto por completo y contar con el entorno necesario, ejecute de manera secuencial los siguientes comandos desde la raíz:

### Windows:

    python -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt

### Linux/MacOS

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

### Ejecucion

#### Autoencoder Básico (Ejercicio 1)

Entrena un autoencoder básico sobre los caracteres del archivo `font.h`:

```bash
python main.py ex1 [latent_dim] [epochs] [noise_level] [deep] [batch_size] [lr] [scale]
```

**Parámetros:**
- `latent_dim`: Tamaño del espacio latente (default: 2)
- `epochs`: Número de épocas de entrenamiento (default: 200)
- `noise_level`: Nivel de ruido para denoising, 0.0 para autoencoder básico (default: 0.0)
- `deep`: `true` para usar capa oculta de 16 neuronas, `false` para arquitectura directa (default: false)
- `batch_size`: Tamaño del batch (default: 4)
- `lr`: Learning rate (default: 0.01)
- `scale`: Escala de datos, `-11` para [-1,1] o `01` para [0,1] (default: -11)

**Ejemplos:**
```bash
# Autoencoder básico con espacio latente 2D
python main.py ex1 2 200 0.0 false 4 0.01 -11

# Autoencoder con capa oculta
python main.py ex1 2 300 0.0 true 4 0.01 -11

# Ejecutar directamente el módulo
python exercises/ej1.py
```

**Salidas generadas:**
- `outputs/autoencoder_loss.png` - Curva de pérdida durante el entrenamiento
- `outputs/autoencoder_originals.png` - Caracteres originales
- `outputs/autoencoder_recon.png` - Caracteres reconstruidos
- `outputs/latent_space.png` - Visualización del espacio latente 2D (si latent_dim=2)
- `outputs/gen_*_*.png` - Caracteres generados por interpolación (si latent_dim=2)
- `outputs/gen_*_generation_report.json` - Reportes de generación

#### Denoising Autoencoder (Ejercicio 1b)

Estudia el comportamiento del autoencoder denoising a diferentes niveles de ruido:

```bash
python main.py denoising
```

O ejecutar directamente:

```bash
python exercises/ej1b_denoising.py
```

**Descripción:**
Este ejercicio implementa un estudio sistemático del autoencoder denoising que:
- Entrena modelos a diferentes niveles de ruido (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
- Evalúa la capacidad de eliminar ruido en cada nivel
- Genera visualizaciones comparativas y curvas de rendimiento
- Documenta la arquitectura elegida y su justificación

**Salidas generadas:**
- `outputs/denoising_comparison_noise_X.XX.png` - Comparación Original | Noisy | Denoised para cada nivel
- `outputs/denoising_loss_noise_X.XX.png` - Curva de pérdida de entrenamiento por nivel
- `outputs/denoising_performance_curves.png` - Curvas de MSE y pixel error vs nivel de ruido
- `outputs/denoising_grid_all_levels.png` - Grid mostrando todos los niveles de ruido
- `outputs/denoising_study_report.json` - Reporte completo con todas las métricas

**Configuración:**
Los parámetros por defecto pueden modificarse editando la función `run()` en `exercises/ej1b_denoising.py`:
- `noise_levels`: Lista de niveles de ruido a estudiar
- `latent_dim`: Dimensión del espacio latente (default: 2)
- `epochs`: Épocas de entrenamiento por nivel (default: 300)
- `deep`: Usar capa oculta (default: True)
- `batch_size`, `lr`, `scale`: Parámetros de entrenamiento