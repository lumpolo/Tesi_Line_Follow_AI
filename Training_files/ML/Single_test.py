import tensorflow as tf

import cv2
import numpy as np
import pandas as pd

import Utils as ut
import Labeling as lb
from Single_test_class import ErrorPredictor

from pathlib import Path

ACTUAL_RUN = "run_gpu_20260924_153226"

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "artifacts" / ACTUAL_RUN / "deploy" / "models" / "final_model.keras"
STATS_PATH = BASE_DIR / "artifacts" / ACTUAL_RUN / "deploy" / "stats" / "target_stats.npz"

CSV_PATH = BASE_DIR / "ML" / "labeled_dataset.csv"


# Controlli preventivi
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Modello non trovato: {MODEL_PATH}")

if not STATS_PATH.exists():
    raise FileNotFoundError(f"Statistiche non trovate: {STATS_PATH}")

tf.keras.mixed_precision.set_global_policy("float32")

# Carica modello e statistiche UNA SOLA VOLTA.
predictor = ErrorPredictor(MODEL_PATH, STATS_PATH)

# Preleva dieci immagini
test_list = ut.img_grab(folders=["ti"])

real_errors = []

absolute_errors = []

predicted_errors = []

useless_examples = 0

for filename, image_path in test_list:
    img = cv2.imread(image_path)

    if img is None:
        print(f"SKIPPED - {filename}: impossibile leggere {image_path}")
        continue

    mask = ut.canonical_mask(img)
    real_error, zone_results = lb.compute_final_error(mask)

    print(f"\nDEBUG - {filename}")
    print(f"Center: {lb.center}")
    print(f"Real error: {real_error}")
    print("Zone results:")

    for zone_name, values in zone_results.items():
        print(f"  {zone_name}: {values}")

    # Immagine senza ground truth valida
    if real_error is None:
        print(f"SKIPPED - {filename}: impossibile calcolare il real error.")
        print("-" * 40)
        useless_examples += 1
        continue

    real_errors.append(real_error)

    result = predictor.evaluate_one(img, real_error)

    
    absolute_errors.append(result["absolute_error"])
    predicted_errors.append(result["predicted_error"])

    print(f"File:            {filename}")
    print(f"Real Error:      {result['real_error']:.2f}")
    print(f"Predicted Error: {result['predicted_error']:.2f}")
    print(f"Absolute Error:  {result['absolute_error']:.2f}")
    print("-" * 40)

print(f"immagini scartate perché error non calcolabile: {useless_examples}")

if not real_errors:
    raise RuntimeError(
        "Nessuna immagine valida: impossibile calcolare le metriche."
    )

real_errors = np.asarray(real_errors, dtype=float)
predicted_errors = np.asarray(predicted_errors, dtype=float)
absolute_errors = np.asarray(absolute_errors, dtype=float)

mean_real_error = float(np.mean(real_errors))
std_real_error = float(np.std(real_errors))

mae = float(np.mean(absolute_errors))
median_absolute_error = float(np.median(absolute_errors))
std_absolute_error = float(np.std(absolute_errors))

rmse = float(np.sqrt(np.mean(
    (predicted_errors - real_errors) ** 2
)))

max_ae = float(np.max(absolute_errors))

ss_res = np.sum(
    (real_errors - predicted_errors) ** 2
)

ss_tot = np.sum(
    (real_errors - np.mean(real_errors)) ** 2
)

r2 = float(
    1 - ss_res / ss_tot
) if ss_tot > 0 else float("nan")

print("\n=== RISULTATI TEST ESTERNO ===")
print(f"Immagini valutate:      {len(real_errors)}")
print(f"Target medio:           {mean_real_error:.2f}")
print(f"Deviazione target:      {std_real_error:.2f}")
print(f"MAE:                    {mae:.2f}")
print(f"Mediana AE:             {median_absolute_error:.2f}")
print(f"Deviazione standard AE: {std_absolute_error:.2f}")
print(f"RMSE:                   {rmse:.2f}")
print(f"Errore massimo:         {max_ae:.2f}")
print(f"R2:                     {r2:.4f}")

print("\n=== DISTRIBUZIONE DELLE PREDIZIONI ===")
print(f"Predizione media: {np.mean(predicted_errors):.2f}")
print(f"Predizione std:   {np.std(predicted_errors):.2f}")
print(f"Predizione min:   {np.min(predicted_errors):.2f}")
print(f"Predizione max:   {np.max(predicted_errors):.2f}")

print("\n=== BIAS ===")
print(
    f"Bias medio predizione - target: "
    f"{np.mean(predicted_errors - real_errors):.2f}"
)

print("\n=== RANGE TEST ===")
print(f"Target min: {np.min(real_errors):.2f}")
print(f"Target max: {np.max(real_errors):.2f}")

print(f"Prediction min: {np.min(predicted_errors):.2f}")
print(f"Prediction max: {np.max(predicted_errors):.2f}")

print("\n=== TRAINING STATS ===")
df = pd.read_csv(CSV_PATH, sep=None, engine="python")
df.columns = [c.strip() for c in df.columns]
print(f"Mean training Dataset: {float(np.mean(df['Final Error']))}")
print(f"Std training Dataset: {float(np.std(df['Final Error']))}")
print(f"Min training Dataset: {float(np.min(df['Final Error']))}")
print(f"Max training Dataset: {float(np.max(df['Final Error']))}")
