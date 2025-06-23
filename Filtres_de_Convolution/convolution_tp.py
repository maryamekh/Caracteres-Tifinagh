import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Flou (moyenne)
blur_kernel = np.ones((3, 3), dtype=np.float32) / 9

# Sobel horizontal
sobel_horizontal = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

# Sobel vertical
sobel_vertical = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float32)

# Netteté
sharpen_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)

# Emboss
emboss_kernel = np.array([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]
], dtype=np.float32)

# Filtres aléatoires
def generate_random_kernel(size, seed=42):
    np.random.seed(seed)
    kernel = np.random.randn(size, size)
    return kernel / np.sum(np.abs(kernel))  



def image_load(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    assert img is not None, "Erreur : image non trouvée."
    if len(img.shape) == 2:
        return img, 'gray'
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, 'rgb'

def display_images(title, images, captions):
    plt.figure(figsize=(18, 5))
    for i, (img, cap) in enumerate(zip(images, captions)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        plt.title(cap)
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()



def convolve_channel(image, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    return output

def apply_convolution(image, kernel):
    if image.ndim == 2:
        output = convolve_channel(image, kernel)
    else:
        output = np.zeros_like(image, dtype=np.float32)
        for c in range(3):
            output[:, :, c] = convolve_channel(image[:, :, c], kernel)
    return np.clip(output, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    image_path = "images/lena.jpg"  
    img, mode = image_load(image_path)

    # Tous les filtres à tester
    filters = {
        "Blur": blur_kernel,
        "Sobel_Horizontal": sobel_horizontal,
        "Sobel_Vertical": sobel_vertical,
        "Sharpen": sharpen_kernel,
        "Emboss": emboss_kernel,
        "Random_5x5": generate_random_kernel(5),
        "Random_7x7": generate_random_kernel(7)
    }

    for name, kernel in filters.items():
        filtered = apply_convolution(img, kernel)
        display_images(f"{name} Filter", [img, filtered], ["Original", name])
        
        # Sauvegarde des résultats
        save_path = f"results/{name.lower()}.jpg"
        if img.ndim == 3:
            filtered = cv2.cvtColor(filtered, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, filtered)
