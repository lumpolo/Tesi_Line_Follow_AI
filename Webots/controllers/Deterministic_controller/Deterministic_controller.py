"""line_follower.py - Controller Webots per e-puck (line following con PID) + debug visivo su Display."""

from controller import Robot        # Import Webots robot API
import numpy as np                  # Import numpy for array operations
from simple_pid import PID          # Import PID controller library
from controller import Display      # Import Display module for debug visualization
from collections import deque       # Import deque for efficient circular buffer

# --- Inizializzazione robot ---
robot = Robot()                             # Create robot instance
print(">>> CONTROLLER AVVIATO <<<")         # Print startup message
timestep = int(robot.getBasicTimeStep())    # Get simulation timestep in milliseconds
print(f">>> timestep = {timestep} <<<")     # Print timestep value

# --- Dispositivi ---
camera = robot.getDevice("camera")          # Get camera device from robot
camera.enable(timestep)                     # Enable camera with timestep refresh

left_motor = robot.getDevice("left wheel motor")    # Get left motor device
right_motor = robot.getDevice("right wheel motor")  # Get right motor device
left_motor.setPosition(float('inf'))                # Set motor to velocity control mode (infinite position)
right_motor.setPosition(float('inf'))               # Set motor to velocity control mode (infinite position)
left_motor.setVelocity(0.0)                         # Initialize left motor velocity to 0
right_motor.setVelocity(0.0)                        # Initialize right motor velocity to 0

# --- Display per debug visivo (opzionali: se non esistono nel .wbt, vengono ignorati) ---
try:
    display_raw = robot.getDevice("display_raw")    # Try to get raw camera display device
except Exception:
    display_raw = None                              # Set to None if device doesn't exist
try:
    display_roi = robot.getDevice("display_roi")    # Try to get ROI/regions display device
except Exception:
    display_roi = None                              # Set to None if device doesn't exist

MAX_SPEED = 5           # Maximum wheel speed in rad/s for e-puck
BASE_SPEED = 2          # Cruise velocity when following line straight
MAX_STEER = BASE_SPEED  # Maximum steering adjustment to prevent reverse motion

# --- Parametri immagine ---
WIDTH = camera.getWidth()       # Get camera image width in pixels
HEIGHT = camera.getHeight()     # Get camera image height in pixels
center_x = WIDTH // 2           # Calculate center x coordinate
center_y = HEIGHT // 2          # Calculate center y coordinate

tolerance_x = max(2, WIDTH // 20)   # Horizontal tolerance for centroid (minimum 2 pixels)
tolerance_y = max(2, HEIGHT // 15)  # Vertical tolerance for centroid (minimum 2 pixels)
offset_y = int(HEIGHT // 4)         # Vertical offset from center for zone positioning

BLACK_THRESHOLD = 60                # Grayscale threshold for line detection (0-255)

# --- Definizione ROI ---
zones = { 
    "top":          (center_y - int(1.5*offset_y) - tolerance_y, center_y - int(1.5*offset_y) + tolerance_y),   # Upper zone
    "center_above": (center_y + int(0.5*offset_y) - tolerance_y, center_y + int(0.5*offset_y)+ tolerance_y),    # Middle zone above
    "center_below": (center_y - int(0.5*offset_y) - tolerance_y, center_y - int(0.5*offset_y) + tolerance_y),   # Middle zone below
    "bottom":       (center_y + int(1.5*offset_y) - tolerance_y, center_y + int(1.5*offset_y) + tolerance_y),   # Lower zone
}
weights = {"top": 0.15, "center_below": 0.25, "center_above": 0.25, "bottom": 0.35}  # Zone weights (bottom zone most important)

# Colors for each zone in debug display (RGB int 0xRRGGBB)
zone_colors = {  
    "top":    0x00A0FF,         # Light blue for top zone
    "center_below": 0xFFFF00,   # Yellow for center below zone
    "center_above": 0x00FF00,   # Green for center above zone
    "bottom": 0xFF9000,         # Orange for bottom zone
}

# --- Parametri anti-flicker ---
LOST_THRESHOLD = 5        # Consecutive frames without line before triggering search
ERROR_SMOOTH_N = 3        # Number of error samples for moving average smoothing

lost_counter = 0  # Counter for frames without detected line
last_error = 0.0  # Last valid error value
error_history = deque(maxlen=ERROR_SMOOTH_N)  # Circular buffer for error smoothing


# --- Funzione per la scala di grigi ---
def get_gray_image():  
    raw = camera.getImage()                                         # Get raw image from camera
    arr = np.frombuffer(raw, np.uint8).reshape((HEIGHT, WIDTH, 4))  # Convert to BGRA array
    gray = arr[:, :, :3].mean(axis=2).astype(np.uint8)              # Convert to grayscale by averaging RGB channels
    return gray, raw                                                # Return both grayscale and raw image


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
            error = cx - center_x                                       # Calculate horizontal error (positive = right, negative = left)
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


# --- Funzione per la visualizzazione della camera ---
def draw_debug(mask, results, used_error):  
    """Draws on two displays: raw camera image + image with ROI/centroids."""
    # --- Display 1: raw camera image ---
    if display_raw is not None:                                                         # If raw display device exists
        img_ref = display_raw.imageNew(camera.getImage(), Display.BGRA, WIDTH, HEIGHT)  # Create image from camera
        display_raw.imagePaste(img_ref, 0, 0, False)                                    # Paste image on display
        display_raw.imageDelete(img_ref)                                                # Delete image reference

    # --- Display 2: binary mask + ROI rectangles + centroids ---
    if display_roi is not None:                                                             # If ROI display device exists
        # Convert binary mask to RGB image (white=line, black=background)
        mask_rgb = np.stack([mask, mask, mask], axis=-1).astype(np.uint8)                   # Stack mask to RGB channels
        mask_bgra = np.dstack([mask_rgb, np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)])    # Add alpha channel
        raw_bytes = mask_bgra.tobytes()                                                     # Convert to bytes for display
        img_ref = display_roi.imageNew(raw_bytes, Display.BGRA, WIDTH, HEIGHT)              # Create image from bytes
        display_roi.imagePaste(img_ref, 0, 0, False)                                        # Paste mask image on display
        display_roi.imageDelete(img_ref)                                                    # Delete image reference

        # Draw ROI rectangles and centroids for each zone
        for name, (y1, y2) in zones.items():                    # Iterate through all zones
            color = zone_colors[name]                           # Get color for this zone
            display_roi.setColor(color)                         # Set drawing color
            display_roi.drawRectangle(0, y1, WIDTH, y2 - y1)    # Draw zone rectangle

            r = results.get(name)                               # Get centroid result for this zone
            if r is not None:                                   # If line was detected in this zone
                cx, cy = r["cx"], r["cy"]                       # Get centroid coordinates
                display_roi.drawLine(cx - 4, cy, cx + 4, cy)    # Draw horizontal crosshair
                display_roi.drawLine(cx, cy - 4, cx, cy + 4)    # Draw vertical crosshair
                display_roi.drawOval(cx, cy, 4, 4)              # Draw circle at centroid

        # Draw vertical lines: image center (error=0 reference) and used error point
        display_roi.setColor(0xFF0000)                                                  # Set color to red for center line
        display_roi.drawLine(center_x, 0, center_x, HEIGHT - 1)                         # Draw center line (error reference)
        used_cx = int(center_x + used_error) if used_error is not None else center_x    # Calculate used error position
        used_cx = max(0, min(WIDTH - 1, used_cx))                                       # Clamp to valid image range
        display_roi.setColor(0xFF00FF)                                                  # Set color to magenta
        display_roi.drawLine(used_cx, 0, used_cx, HEIGHT - 1)                           # Draw line at current error position


# --- Configurazione PID ---
pid = PID(Kp=0.05, Ki=0.0, Kd=0.0, setpoint=0)   # Create PID controller with proportional gain only
pid.output_limits = (-MAX_STEER, MAX_STEER)     # Set output limits for steering
pid.sample_time = timestep / 1000.0             # Set PID sample time

# --- Main control loop ---
while robot.step(timestep) != -1:                           # Loop while simulation is running
    gray, raw = get_gray_image()                            # Capture camera image and convert to grayscale
    mask = (gray < BLACK_THRESHOLD).astype(np.uint8) * 255  # Create binary mask (line = 255)

    final_error, results = compute_final_error(mask)    # Calculate line position error
    line_found = final_error is not None                # Check if line was detected

    if line_found:                          # If line detected
        lost_counter = 0                    # Reset lost counter
        error_history.append(final_error)   # Add error to smoothing buffer
        last_error = final_error            # Update last valid error
    else:                                   # If line not detected
        lost_counter += 1                   # Increment lost counter

    # Error to use: smoothed if found, otherwise last valid until truly lost
    if line_found:                                              # If line detected
        used_error = sum(error_history) / len(error_history)    # Use moving average of errors
    else:                                                       # If line not detected
        used_error = last_error                                 # Use last valid error

    steering = pid(used_error)                              # Calculate steering adjustment using PID controller
    steering = max(-MAX_STEER, min(MAX_STEER, steering))    # Clamp steering to maximum limits

    if lost_counter < LOST_THRESHOLD:       # If line lost for just a few frames
        # Line briefly lost: continue with last valid steering
        left_speed = BASE_SPEED - steering  # Left motor speed (reduced by steering)
        right_speed = BASE_SPEED + steering # Right motor speed (increased by steering)
    else:                                   # If line truly lost (threshold exceeded)
        # Line truly lost: start search by rotating on the spot
        left_speed = MAX_SPEED * 0.3        # Rotate left motor forward
        right_speed = -MAX_SPEED * 0.3      # Rotate right motor backward

    left_speed = float(np.clip(left_speed, -MAX_SPEED, MAX_SPEED))      # Clamp left speed to valid range
    right_speed = float(np.clip(right_speed, -MAX_SPEED, MAX_SPEED))    # Clamp right speed to valid range

    left_motor.setVelocity(left_speed)      # Set left motor velocity
    right_motor.setVelocity(right_speed)    # Set right motor velocity

    draw_debug(mask, results, used_error)  # Draw debug visualization on displays

    print(f"err={used_error:.1f}  steer={steering:.2f}  L={left_speed:.2f} R={right_speed:.2f}  lost={lost_counter}")  # Print debug info
    
