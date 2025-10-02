import sys
import numpy as np


def parse_csv_data(file_path):
    """Reads and processes a CSV file into X_features and Y_labels."""
    x_features = []
    y_labels = []

    try:
        with open(file_path, 'r') as f:
            # --- START CORRECTION: Skip the header line ---
            next(f) # Skips the first line (assumes it's a header/malformed line)
            # ---------------------------------------------

            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = [float(p.strip()) for p in line.split(',') if p.strip()]

                if len(parts) < 4:
                    # Skip rows that don't have enough data (x1, x2, x3, y)
                    continue

                y_labels.append(parts[-1])
                x_features.append(parts[:-1])

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    except StopIteration:
        # Handles case where the file is empty after skipping header
        print("Warning: CSV file is empty after skipping the header.")
        pass
    except ValueError as e:
        # Catch errors if a data cell is not a number
        print(f"Error converting data to float in line: {line.strip()}. Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during file processing: {e}")
        sys.exit(1)

    return x_features, y_labels

def parse_digits_7x5_txt(file_path: str):
    """
    - Etiquetas se infieren como 0..9 repetidos.

    Devuelve (X, labels) donde X tiene shape (N, 35) y labels es (N,) en 0..9.
    """

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_lines = [ln.strip() for ln in f]

        # Filtrar líneas vacías; agrupar de a 7
        content_lines = [ln for ln in raw_lines if ln != ""]
        if len(content_lines) % 7 != 0:
            raise ValueError(
                f"El archivo no contiene un múltiplo de 7 líneas no vacías (actual: {len(content_lines)})."
            )

        X = []
        for i in range(0, len(content_lines), 7):
            block = content_lines[i:i + 7]
            flat = []
            for row in block:
                tokens = row.split()
                if len(tokens) != 5:
                    raise ValueError(f"Fila con {len(tokens)} columnas (se esperaban 5): '{row}'")
                for tok in tokens:
                    if tok not in ("0", "1"):
                        raise ValueError(f"Token inválido (se esperaba 0/1): '{tok}' en fila '{row}'")
                    flat.append(float(tok))
            if len(flat) != 35:
                raise ValueError("Bloque inválido: no se obtuvieron 35 valores.")
            X.append(flat)

        if not X:
            raise ValueError("No se pudieron parsear dígitos del archivo de datos.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)

    X = np.array(X, dtype=int)
    n = X.shape[0]
    labels = np.array([idx % 10 for idx in range(n)], dtype=int)
    return X, labels
