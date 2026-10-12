import cv2
import os
import numpy as np
import pandas as pd

import Utils as ut

# --- Costanti e Parametri ---
HEIGHT = 224
WIDTH = 224
BLACK_THRESHOLD = 60                # Grayscale threshold for line detection (0-255)
BASE = os.path.dirname(os.path.abspath(__file__))
FOLDERS = ["Dataset"]


center = (WIDTH // 2, HEIGHT // 2)
offset = int(HEIGHT // 4)
tolerance_x = max(2, WIDTH // 20)   # Horizontal tolerance for centroid (minimum 2 pixels)
tolerance_y = max(2, HEIGHT // 15)  # Vertical tolerance for centroid (minimum 2 pixels)

# --- Definizione ROI ---
zones = { 
    "top":          (center[1] - int(1.5*offset) - tolerance_y, center[1] - int(1.5*offset) + tolerance_y),   # Upper zone
    "center_above": (center[1] + int(0.5*offset) - tolerance_y, center[1] + int(0.5*offset)+ tolerance_y),    # Middle zone above
    "center_below": (center[1] - int(0.5*offset) - tolerance_y, center[1] - int(0.5*offset) + tolerance_y),   # Middle zone below
    "bottom":       (center[1] + int(1.5*offset) - tolerance_y, center[1] + int(1.5*offset) + tolerance_y),   # Lower zone
}
weights = {"top": 0.15, "center_below": 0.25, "center_above": 0.25, "bottom": 0.35}  # Zone weights (bottom zone most important)


# --- Funzione per il calcolo del centroide (baricentro) ---
def find_centroid(mask_roi):  
    ys, xs = np.where(mask_roi > 0) # Get all pixel coordinates where mask is non-zero
    if len(xs) == 0:                # If no pixels found
        return None                 # Return None (no line detected)
    cx = int(xs.mean())             # Calculate mean x coordinate of line pixels
    cy = int(ys.mean())             # Calculate mean y coordinate of line pixels
    return (cx, cy)                 # Return centroid coordinates

# --- Funzione per il calcolo del centroide (baricentro) ---
def compute_final_error(mask):  
    results = {}                                                        # Dictionary to store results for each zone
    for name, (y1, y2) in zones.items():                                # Iterate through each zone (top, center, bottom)
        y1c, y2c = max(0, y1), min(mask.shape[0], y2)                   # Clip zone boundaries to valid image range
        roi_mask = mask[y1c:y2c, :]                                     # Extract region of interest for this zone
        centroid = find_centroid(roi_mask)                              # Find centroid of line in this zone
        if centroid is not None:                                        # If line was detected in this zone
            cx, cy_local = centroid                                     # Get local centroid coordinates
            cy_full = cy_local + y1c                                    # Convert local y to full image coordinates
            error = cx - center[1]                                      # Calculate horizontal error (positive = right, negative = left)
            results[name] = {"cx": cx, "cy": cy_full, "error": error}   # Store results for this zone
        else:
            results[name] = None                                        # Mark zone as "no line detected"

    total_error = 0.0                                                       # Accumulator for weighted error sum
    total_weight = 0.0                                                      # Accumulator for total weight
    for name, r in results.items():                                         # Iterate through all zone results
        if r is not None:                                                   # If line was detected in this zone
            total_error += r["error"] * weights[name]                       # Add weighted error
            total_weight += weights[name]                                   # Add weight
    final_error = total_error / total_weight if total_weight > 0 else None  # Calculate weighted average error

    return final_error, results                                             # Return final error and all zone results

def label(items=None):
    rows = []  
    if items is None:
        items = ut.img_grab(folders = FOLDERS)
    for idx, (filename, image_path) in enumerate(items):
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if img is None:
            print(f"[SKIP] Impossibile caricare: {image_path}")
            continue

        print(
            f"Immagine {idx}: {filename} | "
            f"shape={img.shape}, dtype={img.dtype}"
        )

        mask = ut.canonical_mask(img)
        final_error, results = compute_final_error(mask)


        print(f"Image {idx}: Final Error = {final_error}")
        rows.append({
            "Id": idx,
            "Image": filename,
            "Final Error": final_error,
            "Top Zone": results.get("top"),
            "Center Above Zone": results.get("center_above"),
            "Center Below Zone": results.get("center_below"),
            "Bottom Zone": results.get("bottom"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(BASE, "labeled_dataset.csv"), index=False)                 # <-- altrimenti perdi tutto
    print(df.head())
    return df
                                              # Append results to DataFrame


if __name__ == "__main__":
    label()
