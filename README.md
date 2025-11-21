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

```bash
python main.py [opciones]
```