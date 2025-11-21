import re
import numpy as np
from pathlib import Path
from typing import Tuple


def _extract_font_ints(text: str):
    # Captura valores escritos como 0x.. o números decimales
    raw = re.findall(r"0x[0-9A-Fa-f]+|\d+", text)
    return [int(token, 0) for token in raw]


def parse_font_h(path: str = "data/font.h", scale: str = "01") -> Tuple[np.ndarray, np.ndarray]:
    """
    Lee el archivo font.h y devuelve:
      X: matriz (n_chars, 35) con los bits del patrón 7x5 aplanados.
      labels: índices 0..n_chars-1

    scale:
      - "01": devuelve bits en {0,1}
      - "-11": remapea a {-1,1}
    """
    data = Path(path).read_text(encoding="utf-8", errors="ignore")
    import re as _re
    # Extraer solo el cuerpo del array Font3 para evitar números fuera del bloque
    m = _re.search(r"Font3\s*\[\s*\d+\s*\]\s*\[\s*\d+\s*\]\s*=\s*\{(.*?)\};", data, flags=_re.S)
    body = m.group(1) if m else data
    # Quitar comentarios y espacios
    data_no_comments = _re.sub(r"//.*", "", body)
    ints = _extract_font_ints(data_no_comments)
    if len(ints) % 7 != 0:
        raise ValueError(f"Font data corrupta: se esperaban múltiplos de 7 filas, obtuve {len(ints)} valores.")

    chars = np.array(ints, dtype=int).reshape(-1, 7)
    bit_rows = []

    for char_rows in chars:
        bits = []
        for val in char_rows:
            row_bits = [0] * 5
            for col in range(5):
                bit = (val >> col) & 1
                row_bits[4 - col] = bit
            bits.extend(row_bits)
        bit_rows.append(bits)

    X = np.array(bit_rows, dtype=float)
    if scale == "-11":
        X = X * 2.0 - 1.0

    labels = np.arange(len(X))
    return X, labels
