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

# ----- lector del archivo 7x5 -----
def parse_digits_7x5_txt(path: str):
    """
    Lee data/TP3-ej3-digitos.txt donde cada dígito son 7 líneas de 5 píxeles (0/1).
    Devuelve:
      X: (N, 35)  float
      labels: (N,) enteros 0..9 asumiendo orden cíclico; si hay más de 10, usa i % 10.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    num_lines_per_digit = 7
    pixels_per_line = 5
    num_digits = len(lines) // num_lines_per_digit

    arr = []
    for i in range(num_digits):
        digit_lines = lines[i * num_lines_per_digit : (i + 1) * num_lines_per_digit]
        # aplanar 7x5 -> 35
        pixels = [int(ch) for line in digit_lines for ch in line.split()]
        if len(pixels) != num_lines_per_digit * pixels_per_line:
            raise ValueError(f"Fila {i}: esperados 35 píxeles, encontrados {len(pixels)}")
        arr.append(pixels)

    X = np.array(arr, dtype=float)
    labels = (np.arange(num_digits) % 10).astype(int)
    return X, labels