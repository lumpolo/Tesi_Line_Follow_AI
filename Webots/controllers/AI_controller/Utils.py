import cv2
import numpy as np

MODEL_WIDTH = 224
MODEL_HEIGHT = 224


def get_mask(image_bgr):
    """
    Receve a BGR OpenCV and returns a mask of the line.
    The mask is obtained by thresholding the grayscale version 
    of the image using Otsu's method, and then inverting it.

    Parameters
    ----------
    image_bgr : np.ndarray
        Image with shape (H, W, 3), in BGR format.

    Returns
    -------
    np.ndarray
        Binary mask with shape (H, W), values 0 or 255.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) # Convert to grayscale

    _, mask = cv2.threshold(                           # Apply Otsu's thresholding
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return mask


def adjust_camera_image(image_bgr, masked=False):
    """
    Preprocess the input image for the ONNX model.

    Pipeline:
      1. Resize to 224 x 224 using INTER_NEAREST.
      2. Apply Otsu mask if masked=True.
      3. Convert BGR to RGB (or grayscale mask to RGB).
      4. Normalize [0, 255] -> [-1, 1].

    Returns
    -------
    np.ndarray
        Image float32, shape (224, 224, 3), interval [-1, 1].
    """

    resized = cv2.resize(                                           # Resize the image to the model's expected input size
        image_bgr,
        (MODEL_WIDTH, MODEL_HEIGHT),
        interpolation=cv2.INTER_NEAREST
    )

    if masked:                                                      # If masked is True, apply Otsu's mask to the resized image
        processed = get_mask(resized)
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)     # Convert the grayscale mask to RGB
    else:
        processed = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)        # Convert the resized BGR image to RGB

    normalized = processed.astype(np.float32)                       # Normalize the image to the range [-1, 1]
    normalized /= 127.5                                             # Scale the values to the range [0, 1]
    normalized -= 1.0                                               # Shift the values to the range [-1, 1]

    return normalized


def get_onnx_input_layout(session):
    """
    Determines the input layout required by the ONNX model.

    Returns
    -------
    str
        'NHWC' per (N, H, W, C)
        'NCHW' per (N, C, H, W)
    """
    input_shape = session.get_inputs()[0].shape

    if len(input_shape) != 4:
        raise ValueError(
            "The ONNX model must receive a 4-dimensional tensor. "
            f"Detected shape: {input_shape}"
        )

    # PyTorch exported models: (N, 3, H, W)
    if input_shape[1] == 3:
        return "NCHW"

    # TensorFlow/Keras exported models: (N, H, W, 3)
    if input_shape[-1] == 3:
        return "NHWC"

    raise ValueError(
        "Impossible to recognize the ONNX input layout. "
        f"Detected shape: {input_shape}"
    )


def apply_onnx_model(session, image):
    """
    Executes a prediction with the ONNX model.

    The final_model.onnx file already includes the destandardization:
        output = output_standardizzato * sigma_y + mu_y

    Quindi il valore restituito e' gia' l'errore finale
    nella scala originale del dataset.
    """
    input_name = session.get_inputs()[0].name                       # Get the name of the input node of the ONNX model
    layout = get_onnx_input_layout(session)                         # Get the input layout of the ONNX model (NHWC or NCHW)

    # (224, 224, 3) -> (1, 224, 224, 3)
    input_data = np.expand_dims(image, axis=0).astype(np.float32)

    # Support for eventual NCHW models.
    if layout == "NCHW":
        input_data = np.transpose(input_data, (0, 3, 1, 2))

    prediction = session.run(
        None,
        {input_name: input_data}
    )[0]

    # The model is a regression model: expected output (1, 1).
    return float(np.squeeze(prediction))

