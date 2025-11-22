"""
Create a simple emoji dataset for VAE training.

This module creates simple binary emoji-like patterns (smiley faces, shapes)
that can be used to train a VAE.
"""

import numpy as np
from typing import Tuple


def create_simple_emoji(size: Tuple[int, int] = (16, 16)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a simple emoji dataset with binary patterns.
    
    Args:
        size: Image size (height, width)
        
    Returns:
        Tuple (X, labels) where:
        - X: Array of shape (n_samples, height * width) with values in [0, 1]
        - labels: Array of labels (0, 1, 2, ...)
    """
    h, w = size
    n_pixels = h * w
    emojis = []
    labels = []
    
    # Helper to create a pattern
    def create_pattern(pattern_func):
        img = np.zeros((h, w), dtype=float)
        for i in range(h):
            for j in range(w):
                if pattern_func(i, j, h, w):
                    img[i, j] = 1.0
        return img.flatten()
    
    # 1. Smiley face
    def smiley(i, j, h, w):
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 3
        dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
        # Face circle
        if abs(dist - radius) < 1.5:
            return True
        # Eyes
        if abs(i - (center_y - radius//3)) < 1 and abs(j - (center_x - radius//2)) < 1:
            return True
        if abs(i - (center_y - radius//3)) < 1 and abs(j - (center_x + radius//2)) < 1:
            return True
        # Smile (arc)
        if i > center_y and abs(np.sqrt((i - center_y)**2 + (j - center_x)**2) - radius*0.7) < 1:
            return True
        return False
    
    # 2. Circle
    def circle(i, j, h, w):
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 3
        dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
        return abs(dist - radius) < 1.5
    
    # 3. Square
    def square(i, j, h, w):
        center_y, center_x = h // 2, w // 2
        size = min(h, w) // 3
        return (abs(i - center_y) < size and abs(j - center_x) < size and
                (abs(i - center_y) == size or abs(j - center_x) == size))
    
    # 4. Triangle
    def triangle(i, j, h, w):
        center_y, center_x = h // 2, w // 2
        size = min(h, w) // 3
        # Top point
        if i == center_y - size and j == center_x:
            return True
        # Left side
        if abs(i - (center_y + size)) < 1 and abs(j - (center_x - size)) < 1:
            return True
        # Right side
        if abs(i - (center_y + size)) < 1 and abs(j - (center_x + size)) < 1:
            return True
        # Base
        if i == center_y + size and abs(j - center_x) < size:
            return True
        # Sides (lines)
        if i > center_y - size and i < center_y + size:
            slope = size / size
            expected_j_left = center_x - (i - (center_y - size)) * slope
            expected_j_right = center_x + (i - (center_y - size)) * slope
            if abs(j - expected_j_left) < 1 or abs(j - expected_j_right) < 1:
                return True
        return False
    
    # 5. Heart
    def heart(i, j, h, w):
        center_y, center_x = h // 2, w // 2
        size = min(h, w) // 4
        # Top curves
        if (i - (center_y - size*1.5))**2 + (j - (center_x - size))**2 < (size*0.8)**2:
            return True
        if (i - (center_y - size*1.5))**2 + (j - (center_x + size))**2 < (size*0.8)**2:
            return True
        # Bottom point
        if i > center_y and abs(j - center_x) < (i - center_y) * 0.8:
            return True
        return False
    
    # 6. Star
    def star(i, j, h, w):
        center_y, center_x = h // 2, w // 2
        size = min(h, w) // 3
        # Center
        if abs(i - center_y) < 1 and abs(j - center_x) < 1:
            return True
        # Points
        for angle in [0, 72, 144, 216, 288]:
            rad = np.deg2rad(angle - 90)
            y = center_y + int(size * np.sin(rad))
            x = center_x + int(size * np.cos(rad))
            if abs(i - y) < 1 and abs(j - x) < 1:
                return True
        return False
    
    # 7. Diamond
    def diamond(i, j, h, w):
        center_y, center_x = h // 2, w // 2
        size = min(h, w) // 3
        return abs(i - center_y) + abs(j - center_x) < size
    
    # 8. Cross
    def cross(i, j, h, w):
        center_y, center_x = h // 2, w // 2
        size = min(h, w) // 3
        return (abs(i - center_y) < size and abs(j - center_x) < 1) or \
               (abs(j - center_x) < size and abs(i - center_y) < 1)
    
    patterns = [smiley, circle, square, triangle, heart, star, diamond, cross]
    
    for label, pattern_func in enumerate(patterns):
        emoji = create_pattern(pattern_func)
        emojis.append(emoji)
        labels.append(label)
    
    # Create multiple variations by adding slight noise/rotations
    # For now, just return the base patterns
    X = np.array(emojis, dtype=float)
    labels = np.array(labels, dtype=int)
    
    return X, labels


def create_emoji_dataset_variations(size: Tuple[int, int] = (16, 16), 
                                   variations_per_class: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create emoji dataset with variations for better training.
    
    Args:
        size: Image size (height, width)
        variations_per_class: Number of variations per emoji class
        
    Returns:
        Tuple (X, labels) with more samples
    """
    h, w = size
    X_base, labels_base = create_simple_emoji(size)
    
    X_all = []
    labels_all = []
    
    for i, (emoji, label) in enumerate(zip(X_base, labels_base)):
        X_all.append(emoji)
        labels_all.append(label)
        
        # Add variations with small perturbations
        for v in range(variations_per_class - 1):
            emoji_var = emoji.copy().reshape(h, w)
            
            # Small random shifts
            shift_y = np.random.randint(-1, 2)
            shift_x = np.random.randint(-1, 2)
            emoji_shifted = np.roll(emoji_var, (shift_y, shift_x), axis=(0, 1))
            
            # Small noise
            noise = np.random.random((h, w)) < 0.05
            emoji_var = emoji_shifted.copy()
            emoji_var[noise] = 1 - emoji_var[noise]
            
            X_all.append(emoji_var.flatten())
            labels_all.append(label)
    
    X = np.array(X_all, dtype=float)
    labels = np.array(labels_all, dtype=int)
    
    return X, labels


if __name__ == "__main__":
    # Test the dataset creation
    X, labels = create_emoji_dataset_variations(size=(16, 16), variations_per_class=4)
    print(f"Dataset shape: {X.shape}")
    print(f"Labels: {np.unique(labels)}")
    print(f"Data range: [{X.min():.2f}, {X.max():.2f}]")
    
    # Visualize a few
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(8):
        ax = axes[i // 4, i % 4]
        ax.imshow(X[i].reshape(16, 16), cmap="gray_r")
        ax.set_title(f"Label {labels[i]}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/emoji_dataset_sample.png", dpi=140)
    print("Sample saved to outputs/emoji_dataset_sample.png")

