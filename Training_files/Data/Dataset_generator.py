import os
import cv2
import random
import numpy as np

def sample_x(w):
    if random.random() < 0.7:
        return int(np.clip(np.random.normal(w/2, w/6), 0, w))
    else:
        return random.randint(0, w)

def generate_augmented_line_images(num_samples=1000, img_size=(224, 224), output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "Dataset")

    os.makedirs(output_dir, exist_ok=True)
    h, w = img_size

    for i in range(num_samples):

        # 1. Background
        base_brightness = random.randint(230, 255)
        img = np.ones((h, w, 3), dtype=np.uint8) * base_brightness

        thickness = random.randint(4, 20)
        line_color = random.randint(10, 50)

        # ---------------------------
        # TIPO DI FORMA
        # ---------------------------
        mode = random.random()

        # ===========================
        # 1. LINEA NORMALE / INVERSA
        # ===========================
        if mode < 0.4:
            x_start = sample_x(w)
            x_mid = sample_x(w)
            x_end = sample_x(w)

            y_start, y_end = (0, h) if random.random() < 0.5 else (h, 0)

            pts = np.array([
                [x_start, y_start],
                [x_mid, h // 2],
                [x_end, y_end]
            ], np.int32)

            cv2.polylines(img, [pts], False,
                          (line_color, line_color, line_color),
                          thickness)

        # ===========================
        # 3. LINEA A "L" (90°)
        # ===========================
        elif mode < 0.9:
            x = sample_x(w)
            y = random.randint(0, h)

            if random.random() < 0.5:
                pts = np.array([
                    [x, 0],
                    [x, y],
                    [random.randint(0, w), y]
                ], np.int32)
            else:
                pts = np.array([
                    [0, y],
                    [x, y],
                    [x, h]
                ], np.int32)

            cv2.polylines(img, [pts], False,
                          (line_color, line_color, line_color),
                          thickness)

        # ---------------------------
        # AUGMENTATION (come tuo)
        # ---------------------------

        if random.random() > 0.5:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)

        if random.random() > 0.3:
            noise = np.random.normal(0, random.randint(10, 30), img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)

        if random.random() > 0.6:
            alpha = random.uniform(0.8, 1.2)
            beta = random.randint(-20, 20)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # ---------------------------
        # SAVE
        # ---------------------------
        cv2.imwrite(
            os.path.join(output_dir, f"line_aug_{i:04d}.png"),
            img
        )

    print(f"Salvate {num_samples} immagini in '{output_dir}'")


if __name__ == "__main__":
    generate_augmented_line_images(num_samples=10000)