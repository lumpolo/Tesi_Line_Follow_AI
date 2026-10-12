import cv2
import os
import numpy as np

# --- Funzione per la scala di grigi ---
def get_mask(image):  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask

def canonical_mask(img):
    if img is None:
        return None

    img_224 = cv2.resize(
        img,
        (224, 224),
        interpolation=cv2.INTER_NEAREST,
    )

    return get_mask(img_224)


def preprocess(img):
    mask = canonical_mask(img)

    if mask is None:
        raise ValueError("Immagine non valida.")

    mask = (mask / 255.0) * 2.0 - 1.0

    return np.stack([mask, mask, mask], axis=-1).astype(np.float32)

def normalize_error(error, min_error=-112, max_error=112):
    """Normalizza l'errore in un range [-1, 1] per la regressione."""
    if error is None:
        return None  # Return None for no line detected
    normalized = (error - min_error) / (max_error - min_error) * 2 - 1
    return np.clip(normalized, -1, 1)

# --- Funzioni per recuperare le immagini ---
def img_grab(base_dir=None, quantity="all", folders = None):
    if folders is None:
        folders=["Dataset", "ti"]
    
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    image_paths = []

    for folder in folders:
        dataset_dir = os.path.join(base_dir, "..", "Data", folder)

        for filename in sorted(os.listdir(dataset_dir)):
            if filename.lower().endswith(".png"):
                image_paths.append(
                    (filename, os.path.join(dataset_dir, filename))
                )

        if quantity != "all":
            image_paths = image_paths[:quantity]

    return image_paths
