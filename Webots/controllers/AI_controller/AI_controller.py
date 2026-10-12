"""
line_follower.py

Controller Webots per e-puck:
- legge il frame della camera;
- converte BGRA Webots -> BGR OpenCV;
- prepara il frame per il modello ONNX;
- predice l'errore laterale;
- usa un PID per comandare le ruote.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from controller import Robot, Display

import Utils as ut


# ------------------------------------------------------------------
# Percorsi
# ------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "final_model.onnx"

# ------------------------------------------------------------------
# Parametri di guida
# ------------------------------------------------------------------

MAX_SPEED = 5.0
BASE_SPEED = 2.0
MAX_STEER = 2.0

KP = 0.05
KI = 0.00
KD = 0.00


# ------------------------------------------------------------------
# Funzioni di supporto
# ------------------------------------------------------------------

def webots_image_to_bgr(raw_image, width, height):
    """
    Converte un buffer immagine Webots BGRA in un array OpenCV BGR.

    Webots Camera.getImage() restituisce byte nel formato:
        BGRA BGRA BGRA ...

    Returns
    -------
    np.ndarray
        Array uint8 BGR con shape (height, width, 3).
    """
    bgra = np.frombuffer(raw_image, dtype=np.uint8)
    bgra = bgra.reshape((height, width, 4))

    # Le prime tre componenti del formato BGRA sono gia' B, G, R.
    bgr = bgra[:, :, :3]

    return bgr


def draw_debug(display_raw, raw_image, width, height):
    """
    Visualizza il frame raw sul Display Webots, se il dispositivo esiste.
    """
    if display_raw is None:
        return

    image_ref = display_raw.imageNew(
        raw_image,
        Display.BGRA,
        width,
        height
    )

    display_raw.imagePaste(image_ref, 0, 0, False)
    display_raw.imageDelete(image_ref)


# ------------------------------------------------------------------
# Inizializzazione Webots
# ------------------------------------------------------------------

robot = Robot()

print(">>> CONTROLLER AVVIATO <<<")

timestep = int(robot.getBasicTimeStep())
print(f">>> timestep = {timestep} ms <<<")

# Camera
camera = robot.getDevice("camera")
camera.enable(timestep)

WIDTH = camera.getWidth()
HEIGHT = camera.getHeight()

print(f">>> Camera: {WIDTH} x {HEIGHT} <<<")

# Motori
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Display opzionale.
# Nel mondo Webots deve esistere un dispositivo chiamato "display_raw".
display_raw = robot.getDevice("display_raw")


# ------------------------------------------------------------------
# Caricamento modello ONNX e statistiche target
# ------------------------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"File modello ONNX non trovato:\n{MODEL_PATH}"
    )

# Carica il modello una sola volta: NON dentro il while.
session = ort.InferenceSession(MODEL_PATH.as_posix())

model_input = session.get_inputs()[0]

print(">>> Modello ONNX caricato <<<")
print(f">>> Input ONNX - nome:  {model_input.name}")
print(f">>> Input ONNX - shape: {model_input.shape}")
print(f">>> Input ONNX - tipo:  {model_input.type}")


# ------------------------------------------------------------------
# Ciclo di controllo
# ------------------------------------------------------------------

while robot.step(timestep) != -1:
    # 1. Acquisizione dell'immagine dalla camera Webots.
    # raw e' un buffer bytes in formato BGRA.
    raw = camera.getImage()

    if raw is None:
        continue

    # 2. BGRA Webots -> BGR OpenCV.
    frame_bgr = webots_image_to_bgr(
        raw_image=raw,
        width=WIDTH,
        height=HEIGHT
    )

    # 3. Preprocessing coerente con il training:
    # maschera -> resize 224 x 224 -> [-1, 1] -> 3 canali.
    model_image = ut.adjust_camera_image(frame_bgr)

    # 4. Inferenza e destandardizzazione.
    error = ut.apply_onnx_model(
        session=session,
        image=model_image
    )

    # 5. PID: il valore desiderato dell'errore e' zero.
    # Segno negativo: se error e' positivo, la linea e' a destra
    # e il robot deve sterzare a destra.
    steering = float(np.clip(
        -KP * error,
        -MAX_STEER,
        MAX_STEER
    ))

    # 6. Controllo differenziale delle ruote.
    #
    # Se steering > 0:
    #   ruota sinistra piu' lenta
    #   ruota destra piu' veloce
    #
    # Se gira dal lato opposto rispetto a quello desiderato,
    # inverti i segni nelle due righe seguenti.
    left_speed = BASE_SPEED - steering
    right_speed = BASE_SPEED + steering

    # 7. Rispetta i limiti del motore.
    left_speed = float(np.clip(
        left_speed,
        -MAX_SPEED,
        MAX_SPEED
    ))

    right_speed = float(np.clip(
        right_speed,
        -MAX_SPEED,
        MAX_SPEED
    ))

    # 8. Applica le velocita'.
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)

    # 9. Debug opzionale.
    draw_debug(
        display_raw=display_raw,
        raw_image=raw,
        width=WIDTH,
        height=HEIGHT
    )

    print(
        f"error={error:.3f} | "
        f"steering={steering:.3f} | "
        f"left={left_speed:.3f} | "
        f"right={right_speed:.3f}"
    )
