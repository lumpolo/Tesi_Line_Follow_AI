from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

import Utils as ut

# ---------------------------------------------------------------
# Percorsi ancorati alla posizione di QUESTO file, non alla CWD.
# ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "labeled_dataset.csv"
STATS_PATH = BASE_DIR / "target_stats.npz"


def dataset_generator(csv_path=CSV_PATH, img_col="Image", err_col="Final Error"):
    """
    Legge il CSV, abbina le immagini alle etichette e restituisce due array numpy.
    Tutto resta in RAM: nessuna allocazione su GPU.

    Returns
    -------
    X : np.ndarray, shape (N, H, W, 3), float32
    y : np.ndarray, shape (N, 1),       float32
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        vicini = [p.name for p in csv_path.parent.iterdir() if p.suffix == ".csv"]
        raise FileNotFoundError(
            f"CSV non trovato: {csv_path}\nCSV presenti in quella cartella: {vicini}"
        )

    # sep=None + engine="python" -> pandas rileva da solo il separatore (, ; \t)
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]

    if img_col not in df.columns or err_col not in df.columns:
        raise KeyError(
            f"Colonne attese: '{img_col}' e '{err_col}'.\n"
            f"Colonne presenti: {df.columns.tolist()}\n"
            f"Prime righe:\n{df.head()}"
        )

    df = df.dropna(subset=[err_col])
    df = df.drop_duplicates(subset=[img_col], keep="first")
    label_map = dict(zip(df[img_col], df[err_col]))

    img_items = ut.img_grab(folders=["Dataset"])

    imgs, errs = [], []
    for filename, image_path in img_items:
        error = label_map.get(filename)

        if error is None:
            continue

        img = cv2.imread(image_path)

        if img is None:
            continue

        imgs.append(np.asarray(ut.preprocess(img), dtype=np.float32))
        errs.append(float(error))

    if not imgs:
        raise ValueError(
            "Nessun campione abbinato.\n"
            f"  immagini da img_grab(): {len(img_items)}\n"
            f"  righe etichettate:      {len(label_map)}\n"
            f"  esempio nome immagine:  {img_items[0][0] if img_items else 'nessuna'}\n"
            f"  esempio nome nel CSV:   {next(iter(label_map), 'nessuno')}\n"
            "I due formati devono coincidere (estensione, maiuscole, path)."
        )

    X = np.stack(imgs)                                  # (N, H, W, 3)
    y = np.asarray(errs, dtype=np.float32)[:, None]     # (N, 1)
    return X, y


def build_dataset(batch_size=32, val_split=0.2, test_split=0.1, seed=42,
                  standardize=True, stats_path=None):
    X, y = dataset_generator()
    n = len(X)

    idx = np.random.default_rng(seed).permutation(n)
    n_val = int(n * val_split)
    n_test = int(n * test_split)
    val_idx   = idx[:n_val]
    test_idx  = idx[n_val:n_val + n_test]
    train_idx = idx[n_val + n_test:]

    if standardize:
        mu_y = float(y[train_idx].mean())
        sigma_y = float(y[train_idx].std())
        if sigma_y == 0.0:
            raise ValueError("sigma del target = 0: le etichette sono tutte uguali.")
        y = (y - mu_y) / sigma_y
        if stats_path is not None:
            stats_path = Path(stats_path)
            stats_path.parent.mkdir(parents=True, exist_ok=True)

            np.savez(
                stats_path,
                mu=mu_y,
                sigma=sigma_y,
            )
    else:
        mu_y, sigma_y = 0.0, 1.0

    print(f"Totale={n}  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    print(f"Immagini: shape={X.shape[1:]}  range=[{X.min():.3f}, {X.max():.3f}]  "
          f"RAM={X.nbytes / 1e9:.2f} GB")
    print(f"Target:   mu={mu_y:.4f}  sigma={sigma_y:.4f}")

    img_shape = X.shape[1:]   # (224, 224, 3)

    def gather_np(batch_idx):
        # eseguito in Python puro: legge le righe da X/y SENZA copiarli in un tensore
        batch_idx = batch_idx.numpy()
        return X[batch_idx], y[batch_idx]

    def make(ids, shuffle=False):
        ids = np.asarray(ids)
        ds = tf.data.Dataset.from_tensor_slices(ids)   # solo indici
        if shuffle:
            ds = ds.shuffle(buffer_size=len(ids), seed=seed,
                            reshuffle_each_iteration=True)
        ds = ds.batch(batch_size)
        ds = ds.map(
            lambda b: tf.py_function(gather_np, [b], (tf.float32, tf.float32)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        # tf.py_function perde le shape: le ripristiniamo esplicitamente
        def set_shapes(xb, yb):
            xb.set_shape([None, *img_shape])
            yb.set_shape([None, 1])
            return xb, yb
        ds = ds.map(set_shapes, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)

    train_ds = make(train_idx, shuffle=True)
    val_ds   = make(val_idx)
    test_ds  = make(test_idx)

    return train_ds, val_ds, test_ds, (mu_y, sigma_y)


if __name__ == "__main__":
    build_dataset()