import cv2
import numpy as np
import time
from simple_pid import PID

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

center_x = 320
tolerance_x = 30

center_y = 240
tolerance_y = 50
offset_y = 140

lower_black = np.uint8([0, 0, 0])
upper_black = np.uint8([51, 51, 51])

zones = {
    "top":    (center_y - offset_y - tolerance_y, center_y - offset_y + tolerance_y),
    "center": (center_y - tolerance_y,             center_y + tolerance_y),
    "bottom": (center_y + offset_y - tolerance_y,  center_y + offset_y + tolerance_y),
}

weights = {"top": 0.2, "center": 0.3, "bottom": 0.5}


def find_centroid(mask_roi):
    """Calcola il baricentro del nero dentro una ROI della mask. Ritorna (cx, cy) o None."""
    contours, _ = cv2.findContours(mask_roi, 1, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy)


def compute_final_error(mask):
    """Calcola l'errore combinato (pesato) delle 3 zone. Ritorna (final_error, results, dict per disegno)."""
    results = {}
    for name, (y1, y2) in zones.items():
        y1c, y2c = max(0, y1), min(mask.shape[0], y2)
        roi_mask = mask[y1c:y2c, :]
        centroid = find_centroid(roi_mask)
        if centroid is not None:
            cx, cy_local = centroid
            cy_full = cy_local + y1c
            error = cx - center_x
            results[name] = {"cx": cx, "cy": cy_full, "error": error}
        else:
            results[name] = None

    total_error = 0
    total_weight = 0
    for name, r in results.items():
        if r is not None:
            total_error += r["error"] * weights[name]
            total_weight += weights[name]
    final_error = total_error / total_weight if total_weight > 0 else 0

    return final_error, results


# --- Configurazione del PID ---
# setpoint = 0 -> vogliamo che l'errore (linea rispetto al centro) sia zero
pid = PID(Kp=0.5, Ki=0.01, Kd=0.1, setpoint=0)
pid.output_limits = (-100, 100)   # limita l'output di sterzata (adatta ai tuoi motori)
pid.sample_time = 0.02            # aggiorna il PID ogni 20 ms circa

while True:
    ret, frame = cap.read()
    if not ret:
        print("Impossibile leggere il frame dalla webcam")
        break

    mask = cv2.inRange(frame, lower_black, upper_black)

    final_error, results = compute_final_error(mask)

    # Il PID prende in input l'errore attuale e restituisce il comando di correzione
    steering = pid(final_error)

    print(f"Errore combinato: {final_error:.1f}  ->  Comando sterzata: {steering:.1f}")

    for name in ["top", "center", "bottom"]:
        r = results[name]
        if r:
            cv2.circle(frame, (r["cx"], r["cy"]), 5, (255, 255, 255), -1)
            print(f"  {name.upper():6} -> cx={r['cx']}, error={r['error']}")
        else:
            print(f"  {name.upper():6} -> I don't see the line")

    for name, (y1, y2) in zones.items():
        cv2.rectangle(frame, (center_x - tolerance_x, y1), (center_x + tolerance_x, y2), (0, 0, 255), 1)

    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)

    time.sleep(0.005)

    if cv2.waitKey(1) & 0xff == ord('o'):
        break

cap.release()
cv2.destroyAllWindows()
