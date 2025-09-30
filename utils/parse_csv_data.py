import sys


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