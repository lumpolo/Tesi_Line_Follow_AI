import cv2
import numpy as np


MODEL_WIDTH = 224
MODEL_HEIGHT = 224


def get_mask(image_bgr):
    """
    Riceve un'immagine BGR OpenCV e restituisce una maschera binaria
    della linea tramite Otsu + inversione.

    Parameters
    ----------
    image_bgr : np.ndarray
        Immagine con shape (H, W, 3), in formato BGR.

    Returns
    -------
    np.ndarray
        Maschera uint8 con shape (H, W), valori 0 oppure 255.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return mask


def adjust_camera_image(image_bgr):
    """
    Prepara un frame BGR per il modello.

    Pipeline:
      1. Maschera Otsu invertita.
      2. Resize a 224 x 224.
      3. Normalizzazione [0, 255] -> [-1, 1].
      4. Conversione a 3 canali identici.

    Parameters
    ----------
    image_bgr : np.ndarray
        Frame OpenCV BGR, shape (H, W, 3).

    Returns
    -------
    np.ndarray
        Immagine float32, shape (224, 224, 3), intervallo [-1, 1].
    """
    mask = get_mask(image_bgr)

    resized = cv2.resize(
        mask,
        (MODEL_WIDTH, MODEL_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    normalized = resized.astype(np.float32) / 255.0
    normalized = normalized * 2.0 - 1.0

    # Da (224, 224) a (224, 224, 3)
    model_image = np.stack(
        [normalized, normalized, normalized],
        axis=-1
    )

    return model_image.astype(np.float32)


def get_onnx_input_layout(session):
    """
    Determina il layout dell'input richiesto dal modello ONNX.

    Returns
    -------
    str
        'NHWC' per (N, H, W, C)
        'NCHW' per (N, C, H, W)
    """
    input_shape = session.get_inputs()[0].shape

    if len(input_shape) != 4:
        raise ValueError(
            "Il modello ONNX deve ricevere un tensore a 4 dimensioni. "
            f"Shape rilevata: {input_shape}"
        )

    # Modelli PyTorch esportati in ONNX: (N, 3, H, W)
    if input_shape[1] == 3:
        return "NCHW"

    # Modelli TensorFlow/Keras esportati in ONNX: (N, H, W, 3)
    if input_shape[-1] == 3:
        return "NHWC"

    raise ValueError(
        "Impossibile riconoscere il layout dell'input ONNX. "
        f"Shape rilevata: {input_shape}"
    )


def apply_onnx_model(session, image):
    """
    Esegue una predizione con il modello ONNX.

    Il file final_model.onnx include gia' la destandardizzazione:
        output = output_standardizzato * sigma_y + mu_y

    Quindi il valore restituito e' gia' l'errore finale
    nella scala originale del dataset.
    """
    input_name = session.get_inputs()[0].name
    layout = get_onnx_input_layout(session)

    # (224, 224, 3) -> (1, 224, 224, 3)
    input_data = np.expand_dims(image, axis=0).astype(np.float32)

    # Supporto eventuale a modelli NCHW.
    if layout == "NCHW":
        input_data = np.transpose(input_data, (0, 3, 1, 2))

    prediction = session.run(
        None,
        {input_name: input_data}
    )[0]

    # Il modello e' di regressione: output atteso (1, 1).
    return float(np.squeeze(prediction))

