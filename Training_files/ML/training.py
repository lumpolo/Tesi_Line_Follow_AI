import time
from pathlib import Path
import argparse
import datetime
import json
import shutil

import numpy as np
import tensorflow as tf

import dataset as ds
from NN_Class import ErrorRegressor

# tf2onnx e' opzionale: se manca, il training procede
# e viene saltata solo l'esportazione ONNX.
try:
    import tf2onnx
except ImportError:
    tf2onnx = None
    print("tf2onnx non disponibile: l'export ONNX verra' saltato.")


BASE_DIR = Path(__file__).resolve().parent      # /home/thomas/Tesi/ML
ARTIFACTS_DIR = BASE_DIR.parent / "artifacts"   # /home/thomas/Tesi/artifacts


# --------------------------------------------------
# DEVICE
# --------------------------------------------------
def setup_device(mode="auto"):
    gpus = tf.config.list_physical_devices("GPU")

    if mode == "cpu":
        tf.config.set_visible_devices([], "GPU")
        device = "/CPU:0"

    elif mode == "gpu":
        if not gpus:
            raise RuntimeError("GPU non disponibile")
        device = "/GPU:0"

    else:
        device = "/GPU:0" if gpus else "/CPU:0"

    if device.startswith("/GPU"):
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)

        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    return device


# --------------------------------------------------
# STRUTTURA DELLE CARTELLE DI OUTPUT
# --------------------------------------------------
def make_run_dirs(tag, timestamp):
    """
    Crea la struttura degli artifact per un singolo training.

    artifacts/run_<tag>_<timestamp>/
    |
    +-- deploy/                 file necessari per usare il modello
    |   +-- models/             .keras .h5 .onnx
    |   +-- stats/              target_stats.npz
    |
    +-- dev/                    materiale di training, non necessario al deploy
        +-- checkpoints/        best.keras
        +-- logs/               history.csv
        +-- metrics/            metrics.json predictions.npz
        +-- code/               snapshot di NN_Class.py
    """
    run_dir = ARTIFACTS_DIR / f"run_{tag}_{timestamp}"

    paths = {
        "run": run_dir,
        "deploy_models": run_dir / "deploy" / "models",
        "deploy_stats": run_dir / "deploy" / "stats",
        "dev_checkpoints": run_dir / "dev" / "checkpoints",
        "dev_logs": run_dir / "dev" / "logs",
        "dev_metrics": run_dir / "dev" / "metrics",
        "dev_code": run_dir / "dev" / "code",
    }

    for directory in paths.values():
        directory.mkdir(parents=True, exist_ok=True)

    return paths


# --------------------------------------------------
# TRAIN
# --------------------------------------------------
def train(mode="auto"):
    start_time = time.time()

    device = setup_device(mode)
    tag = "gpu" if device.startswith("/GPU") else "cpu"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    paths = make_run_dirs(tag, timestamp)

    print(f"\nArtifact directory:\n{paths['run']}\n")

    # Le statistiche del target servono a denormalizzare l'output,
    # quindi appartengono al deploy.
    stats_path = paths["deploy_stats"] / "target_stats.npz"

    # Snapshot del codice della classe custom usata in questo run.
    shutil.copy2(
        BASE_DIR / "NN_Class.py",
        paths["dev_code"] / "NN_Class.py",
    )

    train_ds, val_ds, test_ds, (mu_y, sigma_y) = ds.build_dataset(
        batch_size=32,
        stats_path=stats_path,
    )

    with tf.device(device):
        model = ErrorRegressor(
            input_shape=(224, 224, 3),
            # Addestramento end-to-end: backbone e testa vengono
            # ottimizzati insieme fin dal primo epoch.
            freeze_backbone=False,
            input_range="none",
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="mse",
            metrics=["mae"],
        )

        # ---- TRAINING END-TO-END ----
        # Un'unica fase: backbone e testa sono addestrati insieme.
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(paths["dev_checkpoints"] / "best.keras"),
                save_best_only=True,
            ),
            tf.keras.callbacks.CSVLogger(
                str(paths["dev_logs"] / "history.csv"),
            ),
        ]

        print("\n=== TRAINING END-TO-END ===")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=100,
            callbacks=callbacks,
        )

    training_minutes = (time.time() - start_time) / 60.0

    # --------------------------------------------------
    # SALVATAGGIO PRINCIPALE
    # --------------------------------------------------
    final_model_path = paths["deploy_models"] / "final_model.keras"
    model.save(final_model_path)

    print(f"\nModello finale salvato in:\n{final_model_path}")

    # --------------------------------------------------
    # VALUTAZIONE
    # Eseguita prima delle esportazioni opzionali, cosi' le metriche
    # vengono salvate anche se .h5 oppure ONNX dovessero fallire.
    # --------------------------------------------------
    evaluate(
        model=model,
        test_ds=test_ds,
        mu_y=mu_y,
        sigma_y=sigma_y,
        tag=tag,
        timestamp=timestamp,
        metrics_dir=paths["dev_metrics"],
        training_minutes=training_minutes,
    )

    # --------------------------------------------------
    # ESPORTAZIONE HDF5 (opzionale)
    # --------------------------------------------------
    final_model_h5_path = paths["deploy_models"] / "final_model.h5"

    try:
        model.save(
            final_model_h5_path,
            include_optimizer=False,
        )
        print(f"Modello H5 salvato in:\n{final_model_h5_path}")

    except Exception as error:
        print(f"Salvataggio H5 non riuscito: {error}")

        # Fallback: salva almeno i pesi.
        weights_path = paths["dev_checkpoints"] / "weights.weights.h5"
        model.save_weights(weights_path)

        print(f"Pesi salvati in:\n{weights_path}")

    # --------------------------------------------------
    # ESPORTAZIONE ONNX (opzionale)
    # --------------------------------------------------
    export_onnx(
        model=model,
        mu_y=mu_y,
        sigma_y=sigma_y,
        models_dir=paths["deploy_models"],
    )

    print(f"\nTraining completato in {training_minutes:.1f} minuti.")
    print(f"Artifact directory:\n{paths['run']}")

    return model, paths["run"]


# --------------------------------------------------
# EXPORT ONNX
# --------------------------------------------------
def export_onnx(model, mu_y, sigma_y, models_dir):
    """
    Esporta il modello in ONNX includendo la denormalizzazione:

        predicted_error = output_normalizzato * sigma_y + mu_y

    L'input atteso e' un batch di immagini gia' preprocessate,
    shape (batch, 224, 224, 3), float32, valori in [-1, 1].
    """
    if tf2onnx is None:
        print("Export ONNX saltato: tf2onnx non installato.")
        return

    onnx_model_path = models_dir / "final_model.onnx"

    input_signature = [
        tf.TensorSpec(
            shape=[None, 224, 224, 3],
            dtype=tf.float32,
            name="image",
        )
    ]

    @tf.function(input_signature=input_signature)
    def onnx_inference(image):
        prediction_normalized = model(image, training=False)

        # Il cast esplicito e' necessario perche' su GPU
        # e' attiva la policy mixed_float16.
        prediction = (
            tf.cast(prediction_normalized, tf.float32)
            * tf.constant(sigma_y, dtype=tf.float32)
            + tf.constant(mu_y, dtype=tf.float32)
        )

        return tf.identity(prediction, name="predicted_error")

    try:
        tf2onnx.convert.from_function(
            onnx_inference,
            input_signature=input_signature,
            opset=13,
            output_path=str(onnx_model_path),
        )
        print(f"Modello ONNX salvato in:\n{onnx_model_path}")

    except Exception as error:
        print(f"Export ONNX non riuscito: {error}")
        print("Il modello .keras resta comunque valido.")


# --------------------------------------------------
# TEST
# --------------------------------------------------
def evaluate(model, test_ds, mu_y, sigma_y, tag, timestamp,
             metrics_dir, training_minutes=None):
    y_true, y_pred = [], []

    for xb, yb in test_ds:
        pred = model.predict(xb, verbose=0)

        y_true.append(yb.numpy().ravel())
        y_pred.append(pred.ravel())

    # Da target standardizzato alla scala originale di Error.
    y_true = np.concatenate(y_true) * sigma_y + mu_y
    y_pred = np.concatenate(y_pred) * sigma_y + mu_y

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    median_ae = float(np.median(np.abs(y_true - y_pred)))
    std_ae = float(np.std(np.abs(y_true - y_pred)))
    max_ae = float(np.max(np.abs(y_true - y_pred)))

    denominator = np.sum((y_true - y_true.mean()) ** 2)

    if denominator > 0:
        r2 = float(
            1
            - np.sum((y_true - y_pred) ** 2) / denominator
        )
    else:
        r2 = float("nan")

    print("\n=== TEST RESULTS ===")
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R2   = {r2:.4f}")
    print(f"Median AE = {median_ae:.4f}")
    print(f"Std AE    = {std_ae:.4f}")
    print(f"Max AE    = {max_ae:.4f}")

    mean_pred = float(np.mean(y_pred))
    std_pred = float(np.std(y_pred))
    min_pred = float(np.min(y_pred))
    max_pred = float(np.max(y_pred))

    print("\n=== PREDICTIONS DISTRIBUTION ===")
    print(f"Mean  = {mean_pred:.4f}")
    print(f"Std   = {std_pred:.4f}")
    print(f"Min   = {min_pred:.4f}")
    print(f"Max   = {max_pred:.4f}")

    print("\n=== BIAS ===")
    bias = float(np.mean(y_pred - y_true))
    print(f"Bias  = {bias:.4f}")

    # Tutte le predizioni del test set.
    predictions_path = metrics_dir / "predictions.npz"

    np.savez(
        predictions_path,
        y_true=y_true,
        y_pred=y_pred,
        mu_y=mu_y,
        sigma_y=sigma_y,
    )

    # Metriche leggibili senza dover aprire file numpy.
    metrics_path = metrics_dir / "metrics.json"

    metrics = {
        "CAP 1": "===RUN METRICS===",
        "device": tag,
        "timestamp": timestamp,
        "CAP 2": "===TEST RESULTS===",
        "mae": mae,
        "median_ae": median_ae,
        "rmse": rmse,
        "r2": r2,
        "std_test": std_ae,
        "max_ae": max_ae,
        "CAP 3": "===PREDICTIONS DISTRIBUTION===",
        "mean_pred": mean_pred,
        "std_pred": std_pred,
        "min_pred": min_pred,
        "max_pred": max_pred,
        "CAP 4": "===BIAS===",
        "bias": bias
    }

    if training_minutes is not None:
        metrics["training_minutes"] = round(float(training_minutes), 2)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nPredizioni salvate in:\n{predictions_path}")
    print(f"Metriche salvate in:\n{metrics_path}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
    )
    args = parser.parse_args()

    train(args.device)