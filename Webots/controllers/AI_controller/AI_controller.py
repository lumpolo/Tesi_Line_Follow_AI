"""
line_follower.py

Controller Webots for e-puck:
- reads the camera frame;
- converts BGRA Webots -> BGR OpenCV;
- prepares the frame for the ONNX model;
- predicts the lateral error;
- uses a PID to control the wheels.
"""

from pathlib import Path

import numpy as np
import time as tm
import csv
import json
import onnxruntime as ort

from controller import Robot, Display

import Utils as ut


# ------------------------------------------------------------------
# Percorsi
# ------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

TYPE_OF_MODEL = {
    "mask_pid": "Mask_PID",
    "mask": "Mask_NoPID",
    "normal": "NoMask_NoPID"
}

MODEL_TYPE = "normal"

MODEL_PATH = BASE_DIR / "model" / TYPE_OF_MODEL.get(MODEL_TYPE) / "final_model.onnx"

LOG_PATH_CSV = BASE_DIR.parent / "Log" / "line_follower_AI.csv"

LOG_PATH_JSON = BASE_DIR.parent / "Log" / "line_follower_AI.json"

# ------------------------------------------------------------------
# Variabili globali
# ------------------------------------------------------------------

MASKED = True           # If True, the model expects a masked image (only the line). If False, the model expects a normal image (the whole scene).
PID_CONTROL = True      # If True, the controller will use a PID to control the wheels. If False, the controller will use the model's output directly to control the wheels.
LOG_CSV = True          # If True, the controller will save the log in a CSV file.
LOG_JSON = True         # If True, the controller will save the log in a JSON file.

TIME = []               # List to store inference times for each frame.
LOG_TIMESTAMP = []      # List to store timestamps for each frame.
LOG_ERROR = []          # List to store prediction errors for each frame.
LOG_STEERING = []       # List to store steering values for each frame.

# ------------------------------------------------------------------
# Parametri di guida
# ------------------------------------------------------------------

MAX_SPEED = 5.0         # Maximum speed of the wheels (rad/s).
BASE_SPEED = 1.5        # Base speed of the wheels (rad/s).
MAX_STEER = 2.0         # Maximum steering angle (rad).

KP = 0.08               
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
    bgra = np.frombuffer(raw_image, dtype=np.uint8)     # Convert the raw image buffer to a numpy array
    bgra = bgra.reshape((height, width, 4))             # Reshape the array to (height, width, 4) to represent the BGRA image

    bgr = bgra[:, :, :3]                                # Extract the first three channels (BGR) from the BGRA image

    return bgr


def draw_debug(display_raw, raw_image, width, height):
    """
    Visualizza il frame raw sul Display Webots, se il dispositivo esiste.
    """
    if display_raw is None:                             # Check if the display device exists
        return

    image_ref = display_raw.imageNew(                   # Create a new image on the display with the raw image data
        raw_image,
        Display.BGRA,
        width,
        height
    )

    display_raw.imagePaste(image_ref, 0, 0, False)      # Paste the image onto the display at position (0, 0) without transparency
    display_raw.imageDelete(image_ref)                  # Delete the image reference to free up memory on the display device


# ------------------------------------------------------------------
# Inizializzazione Webots
# ------------------------------------------------------------------

robot = Robot()                                         # Create an instance of the Robot class to interact with the Webots simulation

print(">>> CONTROLLER AVVIATO <<<")

timestamp = 0                                           # Initialize a timestamp variable to keep track of the simulation time in milliseconds

timestep = int(robot.getBasicTimeStep())                # Get the basic time step of the simulation in milliseconds
print(f">>> timestep = {timestep} ms <<<")

# Camera
camera = robot.getDevice("camera")                      # Get the camera device from the robot
camera.enable(timestep)                                 # Enable the camera with the specified time step to start capturing images

WIDTH = camera.getWidth()                               # Get the width of the camera image in pixels
HEIGHT = camera.getHeight()                             # Get the height of the camera image in pixels  

print(f">>> Camera: {WIDTH} x {HEIGHT} <<<")

# Motors
left_motor = robot.getDevice("left wheel motor")        # Get the left wheel motor device from the robot
right_motor = robot.getDevice("right wheel motor")      # Get the right wheel motor device from the robot   

left_motor.setPosition(float("inf"))                    # Set the position of the left wheel motor to infinity to allow continuous rotation
right_motor.setPosition(float("inf"))                   # Set the position of the right wheel motor to infinity to allow continuous rotation

left_motor.setVelocity(0.0)                             # Set the initial velocity of the left wheel motor to 0.0 rad/s
right_motor.setVelocity(0.0)                            # Set the initial velocity of the right wheel motor to 0.0 rad/s

# Display optional.
# In the world, there must be a device named "display_raw".
display_raw = robot.getDevice("display_raw")            # Get the display device from the robot to visualize the raw camera image (if it exists)   


# ------------------------------------------------------------------
# Caricamento modello ONNX e statistiche target
# ------------------------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(                                # Control of the existence of the ONNX model file
        f"File modello ONNX non trovato:\n{MODEL_PATH}"
    )

session = ort.InferenceSession(MODEL_PATH.as_posix())       # Load the ONNX model into memory and create an inference session

model_input = session.get_inputs()[0]                       # Get the input information of the ONNX model (name, shape, type)

print(">>> Modello ONNX caricato <<<")
print(f">>> Input ONNX - name:  {model_input.name}")
print(f">>> Input ONNX - shape: {model_input.shape}")
print(f">>> Input ONNX - type:  {model_input.type}")


# ------------------------------------------------------------------
# Ciclo di controllo
# ------------------------------------------------------------------

while robot.step(timestep) != -1:
    # 1. Read the camera image.
    raw = camera.getImage()                                                 # raw is a bytes buffer in BGRA format.

    if raw is None:                                                         # If the camera image is not available, skip this iteration of the loop
        continue

    # 2. BGRA Webots -> BGR OpenCV.
    frame_bgr = webots_image_to_bgr(
        raw_image=raw,
        width=WIDTH,
        height=HEIGHT
    )

    # 3. Preprocessing the same with the training phase of the model:
    model_image = ut.adjust_camera_image(frame_bgr, masked=MASKED)          # resize 224 x 224 -> mask -> [-1, 1] -> 3 channels.

    start_time = tm.time()                                                  # Start measuring the inference time for the ONNX model prediction

    # 4. inference.
    error = ut.apply_onnx_model(
        session=session,
        image=model_image
    )

    end_time = tm.time()                                                    # End measuring the inference time for the ONNX model prediction
    inference_time = end_time - start_time                                  # Calculate the inference time in seconds
    TIME.append(inference_time)                                             # Append the inference time to the TIME list for later analysis
    
    timestamp += timestep                                                   # Increment the timestamp by the timestep to keep track of the simulation time in milliseconds
    LOG_TIMESTAMP.append(timestamp)                                         # Append the current timestamp to the LOG_TIMESTAMP list for later analysis
    LOG_ERROR.append(error)                                                 # Append the predicted error to the LOG_ERROR list for later analysis

    # 5. PID.
    if PID_CONTROL:
        dt = timestep / 1000.0                                                  # Convert the timestep from milliseconds to seconds for PID calculations
        integral = sum(LOG_ERROR) * dt                                          # Calculate the integral term of the PID controller
        integral = np.clip(integral, -MAX_STEER, MAX_STEER)                     # Clip the integral term to prevent windup

        if len(LOG_ERROR) > 1:
            derivative = (LOG_ERROR[-1] - LOG_ERROR[-2]) / dt                   # Calculate the derivative term of the PID controller
        else:
            derivative = 0.0

        steering = float(np.clip(
            -KP * error -KI * integral -KD * derivative,                        # Negative because IT WORKS
            -MAX_STEER,
            MAX_STEER
        ))
    else:
        steering = float(np.clip(
            -error,                                                             # Negative because IT WORKS
            -MAX_STEER,
            MAX_STEER
        ))

    LOG_STEERING.append(steering)                                           # Append the calculated steering value to the LOG_STEERING list for later analysis

    # 6. Differential control for wheels.
    left_speed = BASE_SPEED - steering                                      # singns are inverted because IT WORKS
    right_speed = BASE_SPEED + steering

    # 7. Respect the speed limits.
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

    # 8. Apply the speeds.
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)

    # 9. Optional debug.
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

# Save the data in a CSV file
if LOG_CSV:
    with open(LOG_PATH_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp (ms)', 'Error', 'Steering', 'Inference_Time'])
        for i in range(len(LOG_TIMESTAMP)):
            writer.writerow([
                LOG_TIMESTAMP[i],
            LOG_ERROR[i],
            LOG_STEERING[i],
            TIME[i]
        ])

# Save the data in a JSON file
if LOG_JSON:
    log_data = {
        "mean inference_time": float(np.mean(TIME)),
        "std inference_time": float(np.std(TIME)),
        "max positive error": float(np.max(LOG_ERROR)),
        "max negative error": float(np.min(LOG_ERROR)),
        "mean error": float(np.mean(LOG_ERROR))
}

    with open(LOG_PATH_JSON, mode='w') as f:
        json.dump(log_data, f, indent=4)