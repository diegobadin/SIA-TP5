"""
Parse ASCII art emoji dataset from text file.

Format: ASCII art where '#' = 1 and '.' = 0
Each emoji is separated by blank lines.
"""

import numpy as np
from pathlib import Path
from typing import Tuple


def parse_emoji_txt(path: str = "data/emojis.txt", scale: str = "01") -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse ASCII art emoji dataset from text file.
    
    Args:
        path: Path to emoji text file
        scale: "01" for [0,1] or "-11" for [-1,1]
        
    Returns:
        Tuple (X, labels) where:
        - X: Array of shape (n_emojis, height * width) with values in [0,1] or [-1,1]
        - labels: Array of labels (0, 1, 2, ...)
    """
    data = Path(path).read_text(encoding="utf-8", errors="ignore")
    
    # Split by blank lines to separate emojis
    # Each emoji starts with a comment line "# Emoji N: Name" followed by ASCII art
    emoji_blocks = []
    current_block = []
    in_emoji = False
    
    for line in data.split('\n'):
        stripped = line.strip()
        
        # Check if this is an emoji header comment
        if stripped.startswith('# Emoji'):
            # Save previous block if exists
            if current_block:
                emoji_blocks.append(current_block)
            current_block = []
            in_emoji = True
            continue
        
        # Skip other comments and empty lines (but collect data lines)
        if stripped.startswith('#') and not in_emoji:
            continue
        
        if not stripped:
            # Empty line - if we have a block, it's complete
            if current_block:
                emoji_blocks.append(current_block)
                current_block = []
                in_emoji = False
            continue
        
        # Collect data lines (ASCII art)
        if in_emoji and not stripped.startswith('#'):
            current_block.append(stripped)
    
    # Add last block if exists
    if current_block:
        emoji_blocks.append(current_block)
    
    emojis = []
    labels = []
    
    # First pass: determine max dimensions
    max_height = 0
    max_width = 0
    
    for block in emoji_blocks:
        if not block:
            continue
        max_height = max(max_height, len(block))
        max_width = max(max_width, max(len(row) for row in block) if block else 0)
    
    # Use square dimensions (pad to make square if needed)
    size = max(max_height, max_width)
    
    # Second pass: parse and pad to consistent size
    for label, rows in enumerate(emoji_blocks):
        if not rows:
            continue
        # Create binary matrix
        img = np.zeros((size, size), dtype=float)
        for i, row in enumerate(rows):
            if i >= size:
                break
            for j, char in enumerate(row):
                if j >= size:
                    break
                if char == '#':
                    img[i, j] = 1.0
        
        emojis.append(img.flatten())
        labels.append(label)
    
    X = np.array(emojis, dtype=float)
    
    # Normalize to [0, 1] or [-1, 1]
    if scale == "-11":
        X = X * 2.0 - 1.0
    
    labels = np.array(labels, dtype=int)
    return X, labels


if __name__ == "__main__":
    # Test the parser
    X, labels = parse_emoji_txt()
    print(f"Loaded {len(X)} emojis")
    print(f"Shape: {X.shape}")
    print(f"Labels: {labels}")
    print(f"Data range: [{X.min():.2f}, {X.max():.2f}]")
    
    # Visualize
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(min(8, len(X))):
        ax = axes[i // 4, i % 4]
        # Determine image dimensions (assume square for now)
        size = int(np.sqrt(X.shape[1]))
        ax.imshow(X[i].reshape(size, size), cmap="gray_r")
        ax.set_title(f"Emoji {labels[i]}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/emoji_dataset_sample.png", dpi=140)
    print("Sample saved to outputs/emoji_dataset_sample.png")

