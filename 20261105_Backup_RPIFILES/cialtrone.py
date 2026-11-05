#!/usr/bin/env python3
"""
Line follower per Raspberry Pi con Pixy e Arduino.

Acquisizione:
    acquisisci.py -> acquisisci_frames() -> generatore di (rgb, raw)

Protocollo seriale verso Arduino:
    sinistra,destra\\n

Esempi:
    80,80\\n      avanti
    110,50\\n     curva a destra
    -50,50\\n     rotazione verso sinistra
    0,0\\n        arresto

I valori sono PWM firmati, non velocità in rad/s.

Arduino OPZIONALE: se non viene trovato (o cade durante la corsa) lo script
lo segnala e continua lo stesso, producendo log CSV/JSON e immagini.

Avvio:
    python3 line_follow.py                     # porta Arduino rilevata da sola
    python3 line_follow.py --port /dev/ttyACM1 # porta esplicita
    python3 line_follow.py --debug             # finestre OpenCV
    python3 line_follow.py --no-serial         # ignora Arduino di proposito
    python3 line_follow.py --save-images 5     # un'immagine ogni 5 cicli
    python3 line_follow.py --save-images 0     # nessuna immagine

Uscite in Log/: line_follower_Pixy.csv, line_follower_Pixy.json, immagini/*.png

Immagini salvate (ogni N cicli e a ogni cambio di stato):
    frame_XXXXX_STATO.png         il frame a colori
    frame_XXXXX_STATO_debug.png   frame pulito + maschera con ROI, centroidi ed errore
                                  (solo se SALVA_IMMAGINI_DEBUG = True, qui sotto)

Richiede:
    numpy, pyserial, pixycam.py + libpixyframe.so
OpenCV serve solo con --debug.
"""

import argparse
import csv
import json
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import numpy as np
import serial
from serial.tools import list_ports

from acquisisci import acquisisci_frames
from pixycam import PixyError, bayer_to_rgb


# ============================================================
# PERCORSI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "Log"

LOG_PATH_CSV = LOG_DIR / "line_follower_Pixy.csv"
LOG_PATH_JSON = LOG_DIR / "line_follower_Pixy.json"
IMAGE_DIR = LOG_DIR / "immagini"


# ============================================================
# SALVATAGGIO IMMAGINI
# ============================================================

# True  -> oltre al frame salva anche l'immagine di debug: a sinistra il frame
#          pulito, a destra la maschera con ROI, centroidi, errore di ogni ROI
#          ed errore finale.
# False -> salva solo il frame a colori.
# (La frequenza si regola con --save-images N; --save-images 0 non salva nulla.)
SALVA_IMMAGINI_DEBUG = True

# Ingrandimento dell'immagine di debug, per leggere meglio scritte e marcatori.
SCALA_DEBUG = 2


# ============================================================
# PARAMETRI VISIONE
# ============================================================

BLACK_THRESHOLD = 60
ERROR_SMOOTH_N = 3

WEIGHTS = {
    "top": 0.15,
    "center_above": 0.25,
    "center_below": 0.25,
    "bottom": 0.35,
}


# ============================================================
# PARAMETRI CONTROLLO
# ============================================================

BASE_PWM = 80
MAX_PWM = 160
MAX_STEER_PWM = 80
SEARCH_PWM = 50

# Errore normalizzato circa tra -1 e +1; controllo solo P.
KP = 160.0

LOST_GRACE_SECONDS = 0.15
SEARCH_TIMEOUT_SECONDS = 2.0
CONTROL_HZ = 20.0

# Molte schede Arduino si resettano all'apertura della seriale.
ARDUINO_STARTUP_SECONDS = 2.5

# Invertire se un motore gira nel verso sbagliato.
LEFT_SIGN = 1
RIGHT_SIGN = 1


# ============================================================
# RILEVAMENTO PORTA ARDUINO
# ============================================================

# La Pixy (USB vendor, libusb) non deve mai essere scambiata per Arduino.
PIXY_VID_PID = {
    (0x1FC9, 0x000C),  # Pixy (CMUcam5)
    (0xB1AC, 0xF000),  # Pixy2
}

# Costruttori comuni: Arduino, Arduino (2a03), CH340, FTDI, CP210x.
ARDUINO_VIDS = {0x2341, 0x2A03, 0x1A86, 0x0403, 0x10C4}


def find_arduino_port():
    """Cerca una porta seriale che sembri Arduino, escludendo la Pixy."""
    candidates = []

    for p in list_ports.comports():
        if p.vid is None:
            continue
        if (p.vid, p.pid) in PIXY_VID_PID:
            continue
        if p.vid in ARDUINO_VIDS or "arduino" in (p.manufacturer or "").lower():
            candidates.append(p.device)

    return sorted(candidates)[0] if candidates else None


def open_serial(port, baud):
    """Apre la seriale e attende il reset di Arduino."""
    if port == "auto":
        port = find_arduino_port()
        if port is None:
            raise serial.SerialException(
                "Nessun Arduino trovato. Porte viste: "
                + (", ".join(p.device for p in list_ports.comports()) or "nessuna")
                + ". Usa --port oppure --no-serial."
            )

    print(f"Apertura porta seriale {port} a {baud} baud...")
    ser = serial.Serial(port, baud, timeout=0.1, write_timeout=0.2)
    time.sleep(ARDUINO_STARTUP_SECONDS)
    ser.reset_input_buffer()
    return ser


# ============================================================
# SERIALE
# ============================================================

def send_motors(ser, left, right):
    """
    Invia una riga: sinistra,destra\\n
    Se ser è None (--no-serial) calcola solo i valori.
    Restituisce i comandi dopo limitazione e inversione.
    """
    left = int(round(np.clip(LEFT_SIGN * left, -MAX_PWM, MAX_PWM)))
    right = int(round(np.clip(RIGHT_SIGN * right, -MAX_PWM, MAX_PWM)))

    if ser is not None:
        message = f"{left},{right}\n".encode("ascii")
        written = ser.write(message)

        if written != len(message):
            raise IOError("Comando seriale inviato solo parzialmente.")

    return left, right


# ============================================================
# ROI E CENTROIDI
# ============================================================

def make_zones(height):
    """Quattro fasce orizzontali (y cresce verso il basso)."""
    center_y = height // 2
    offset_y = height // 4
    tolerance_y = max(2, height // 15)

    centers = {
        "top": center_y - int(1.5 * offset_y),
        "center_above": center_y - int(0.5 * offset_y),
        "center_below": center_y + int(0.5 * offset_y),
        "bottom": center_y + int(1.5 * offset_y),
    }

    return {
        name: (max(0, cy - tolerance_y), min(height, cy + tolerance_y))
        for name, cy in centers.items()
    }


def find_centroid(mask_roi):
    ys, xs = np.nonzero(mask_roi)

    if xs.size == 0:
        return None

    return float(xs.mean()), float(ys.mean())


def compute_final_error(mask):
    """
    Errore medio pesato delle ROI.
        negativo -> linea a sinistra
        positivo -> linea a destra
    """
    height, width = mask.shape
    center_x = (width - 1) / 2.0

    zones = make_zones(height)
    results = {}

    weighted_error = 0.0
    total_weight = 0.0

    for name, (y1, y2) in zones.items():
        centroid = find_centroid(mask[y1:y2, :])

        if centroid is None:
            results[name] = None
            continue

        cx, cy_local = centroid
        error = (cx - center_x) / max(center_x, 1.0)

        results[name] = {
            "cx": int(round(cx)),
            "cy": int(round(cy_local + y1)),
            "error": float(error),
        }

        weighted_error += error * WEIGHTS[name]
        total_weight += WEIGHTS[name]

    final_error = weighted_error / total_weight if total_weight > 0 else None

    return final_error, results, zones


# ============================================================
# DEBUG GRAFICO
# ============================================================

def draw_debug(cv2, rgb, mask, results, zones, used_error, state):
    """Mostra immagine e maschera. True quando si preme q."""
    frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    height, width = mask.shape
    center_x = (width - 1) / 2.0

    colors = {
        "top": (255, 160, 0),
        "center_above": (0, 255, 0),
        "center_below": (0, 255, 255),
        "bottom": (0, 144, 255),
    }

    for name, (y1, y2) in zones.items():
        color = colors[name]
        cv2.rectangle(debug, (0, y1), (width - 1, y2 - 1), color, 1)

        result = results[name]
        if result is not None:
            cv2.drawMarker(
                debug, (result["cx"], result["cy"]),
                color, cv2.MARKER_CROSS, 10, 1,
            )

    used_x = int(round(center_x + used_error * center_x))
    used_x = max(0, min(width - 1, used_x))

    cv2.line(debug, (int(center_x), 0), (int(center_x), height - 1), (0, 0, 255), 1)
    cv2.line(debug, (used_x, 0), (used_x, height - 1), (255, 0, 255), 1)

    cv2.putText(
        debug, state, (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
    )

    cv2.imshow("Pixy", frame_bgr)
    cv2.imshow("Linea e ROI", debug)

    return (cv2.waitKey(1) & 0xFF) == ord("q")


def save_images_safe(*args):
    """Versione per il thread di salvataggio: non solleva mai."""
    try:
        save_images(*args)
    except Exception as e:
        print(f"\nImpossibile salvare l'immagine: {e}")


ROI_COLORS = {
    "top": (0, 160, 255),
    "center_above": (0, 255, 0),
    "center_below": (255, 255, 0),
    "bottom": (255, 144, 0),
}


def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def save_images(base_path, rgb, mask, results, zones, final_error, used_error, state):
    """
    Salva base_path.png (frame a colori) e, se SALVA_IMMAGINI_DEBUG e' True,
    base_path_debug.png: a sinistra il frame com'e' (senza disegni), a destra
    la maschera con ROI, centroidi, errore di ogni ROI, centro immagine (rosso)
    ed errore usato (magenta).
    Usa solo Pillow: funziona anche senza display e senza OpenCV.
    """
    from PIL import Image, ImageDraw

    base_path = Path(base_path)
    Image.fromarray(rgb).save(base_path.with_suffix(".png"), compress_level=1)

    if not SALVA_IMMAGINI_DEBUG:
        return

    k = max(1, int(SCALA_DEBUG))
    height, width = mask.shape
    w, h = width * k, height * k
    center_x = (width - 1) / 2.0

    used_x = int(round(center_x + used_error * center_x))
    used_x = max(0, min(width - 1, used_x))

    left_img = Image.fromarray(rgb).convert("RGB").resize((w, h), Image.NEAREST)
    right_img = Image.fromarray(mask).convert("RGB").resize((w, h), Image.NEAREST)
    small = _font(10 * k // 2 + 6)

    # i disegni vanno solo sulla maschera: il frame a sinistra resta pulito
    for img in (right_img,):
        d = ImageDraw.Draw(img)

        for name, (y1, y2) in zones.items():
            color = ROI_COLORS[name]
            d.rectangle([0, y1 * k, w - 1, y2 * k - 1], outline=color, width=k)

            r = results[name]
            if r is not None:
                x, y = r["cx"] * k, r["cy"] * k
                d.line([x - 6 * k, y, x + 6 * k, y], fill=color, width=k)
                d.line([x, y - 6 * k, x, y + 6 * k], fill=color, width=k)

        d.line([int(center_x) * k, 0, int(center_x) * k, h - 1], fill=(255, 0, 0), width=k)
        d.line([used_x * k, 0, used_x * k, h - 1], fill=(255, 0, 255), width=k)

    # errore di ogni ROI, scritto sulla maschera accanto alla sua fascia
    d = ImageDraw.Draw(right_img)
    for name, (y1, y2) in zones.items():
        r = results[name]
        label = f"{name}: " + ("--" if r is None else f"{r['error']:+.2f}")
        d.text((4, y1 * k + 2 * k), label, fill=ROI_COLORS[name], font=small,
               stroke_width=2, stroke_fill=(0, 0, 0))

    # barra in alto: stato, errore del frame, errore filtrato usato per sterzare
    bar = 18 * k
    canvas = Image.new("RGB", (w * 2, h + bar), (20, 20, 20))
    canvas.paste(left_img, (0, bar))
    canvas.paste(right_img, (w, bar))
    err_txt = "--" if final_error is None else f"{final_error:+.3f}"
    ImageDraw.Draw(canvas).text(
        (6, 3 * k),
        f"{state}   errore frame = {err_txt}   errore usato (filtrato) = {used_error:+.3f}",
        fill=(255, 255, 255), font=_font(9 * k),
    )
    canvas.save(base_path.with_name(base_path.name + "_debug.png"), compress_level=1)


# ============================================================
# STATISTICHE
# ============================================================

def compute_stats(values):
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}

    arr = np.asarray(values, dtype=float)

    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def save_summary(errors, processing_times, acquisition_times):
    summary = {
        "error_normalized": compute_stats(errors),
        "processing_time_s": compute_stats(processing_times),
        "acquisition_time_s": compute_stats(acquisition_times),
        "configuration": {
            "black_threshold": BLACK_THRESHOLD,
            "error_smooth_n": ERROR_SMOOTH_N,
            "kp": KP,
            "base_pwm": BASE_PWM,
            "max_pwm": MAX_PWM,
            "max_steer_pwm": MAX_STEER_PWM,
            "search_pwm": SEARCH_PWM,
            "control_hz_limit": CONTROL_HZ,
        },
    }

    with LOG_PATH_JSON.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, allow_nan=False)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Line follower Pixy + Arduino")
    parser.add_argument("--port", default="auto",
                        help="porta seriale Arduino (default: auto)")
    parser.add_argument("--baud", type=int, default=9600,
                        help="baud rate: deve coincidere con lo sketch Arduino")
    parser.add_argument("--debug", action="store_true",
                        help="mostra camera e ROI; richiede OpenCV e display")
    parser.add_argument("--no-serial", action="store_true",
                        help="ignora Arduino di proposito (nessun tentativo di apertura)")
    parser.add_argument("--save-images", type=int, default=10, metavar="N",
                        help="salva le immagini ogni N cicli e a ogni cambio di stato "
                             "(0 = disattivato, default 10); l'immagine di debug "
                             "dipende da SALVA_IMMAGINI_DEBUG")
    parser.add_argument("--max-images", type=int, default=300,
                        help="numero massimo di immagini salvate (default 300)")
    args = parser.parse_args()

    cv2 = None
    if args.debug:
        try:
            import cv2
        except ImportError:
            print("Errore: --debug richiede OpenCV (pip install opencv-python).")
            return 1

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Arduino opzionale: se manca lo si segnala e si prosegue (solo log/immagini).
    ser = None
    if not args.no_serial:
        try:
            ser = open_serial(args.port, args.baud)
            print("Arduino connesso.")
        except (serial.SerialException, OSError) as e:
            print(f"ATTENZIONE: Arduino non disponibile ({e})")
            print(">>> Si prosegue SENZA Arduino: nessun comando ai motori, "
                  "vengono prodotti solo log e immagini.")
            ser = None
    else:
        print("--no-serial: Arduino escluso, nessun comando verrà inviato.")

    saved_images = 0
    cycle = 0
    executor = ThreadPoolExecutor(max_workers=1)  # salvataggio PNG fuori dal ciclo
    pending = []
    prev_state = None
    if args.save_images > 0:
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            import PIL  # noqa: F401
        except ImportError:
            print("Pillow non installato: immagini disattivate.")
            args.save_images = 0

    errors_log, proc_times_log, acq_times_log = [], [], []

    error_history = deque(maxlen=ERROR_SMOOTH_N)
    state = "START"
    last_seen_time = time.monotonic()
    search_start_time = None
    last_error = 0.0
    smoothed_error = 0.0

    csv_file = LOG_PATH_CSV.open("w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "t_s", "state", "error", "smoothed_error",
        "left_pwm", "right_pwm", "acq_s", "proc_s",
    ])
    t0 = time.monotonic()
    exit_code = 0

    print("Avvio ciclo di controllo. Ctrl+C per interrompere.")

    try:
        # Il generatore va creato UNA volta: apre la Pixy al primo next()
        # e la richiude con closing().
        with closing(acquisisci_frames(con_rgb=False)) as frames:
            while True:
                loop_start = time.monotonic()

                # 1. Acquisizione
                try:
                    rgb, raw = next(frames)
                except StopIteration:
                    print("Flusso di frame terminato.")
                    break
                acq_time = time.monotonic() - loop_start

                # 2. Elaborazione
                t_proc = time.monotonic()
                # Soglia direttamente sul Bayer grezzo: niente demosaico per frame.
                mask = (raw < BLACK_THRESHOLD).astype(np.uint8) * 255
                final_error, results, zones = compute_final_error(mask)
                proc_time = time.monotonic() - t_proc

                now = time.monotonic()

                # 3. Stati
                if final_error is not None:
                    last_seen_time = now
                    search_start_time = None
                    state = "TRACKING"
                    last_error = final_error
                    error_history.append(final_error)
                    smoothed_error = sum(error_history) / len(error_history)
                else:
                    lost_for = now - last_seen_time

                    if lost_for < LOST_GRACE_SECONDS:
                        state = "LOST_GRACE"
                    else:
                        if search_start_time is None:
                            search_start_time = now
                            error_history.clear()

                        if now - search_start_time < SEARCH_TIMEOUT_SECONDS:
                            state = "SEARCHING"
                        else:
                            state = "STOP_LOST"

                # 4. Controllo motori
                if state == "STOP_LOST":
                    left_pwm, right_pwm = 0, 0
                elif state == "SEARCHING":
                    # Ruota verso il lato in cui la linea è sparita.
                    direction = 1 if last_error >= 0 else -1
                    left_pwm = direction * SEARCH_PWM
                    right_pwm = -direction * SEARCH_PWM
                else:
                    # TRACKING e LOST_GRACE: mantiene l'ultimo errore valido.
                    steer = float(np.clip(
                        KP * smoothed_error, -MAX_STEER_PWM, MAX_STEER_PWM
                    ))
                    left_pwm = BASE_PWM + steer
                    right_pwm = BASE_PWM - steer

                try:
                    left_pwm, right_pwm = send_motors(ser, left_pwm, right_pwm)
                except (serial.SerialException, IOError) as e:
                    print(f"\nATTENZIONE: Arduino perso ({e}). "
                          "Si prosegue senza invio comandi.")
                    try:
                        ser.close()
                    except Exception:
                        pass
                    ser = None
                    left_pwm, right_pwm = send_motors(None, left_pwm, right_pwm)

                # Immagini su disco (periodiche + a ogni cambio di stato)
                if (args.save_images > 0 and saved_images < args.max_images
                        and (cycle % args.save_images == 0 or state != prev_state)):
                    pending[:] = [f for f in pending if not f.done()]
                    if len(pending) < 4:  # se il disco e' lento, salta invece di rallentare il ciclo
                        pending.append(executor.submit(
                            save_images_safe,
                            IMAGE_DIR / f"frame_{cycle:05d}_{state}",
                            bayer_to_rgb(raw), mask, results, zones,
                            final_error, smoothed_error, state,
                        ))
                        saved_images += 1
                prev_state = state
                cycle += 1

                if cycle % 50 == 0:
                    dt = time.monotonic() - t0
                    print(f"\r{cycle / dt:.1f} cicli/s | acquisizione "
                          f"{1000 * np.mean(acq_times_log[-50:]):.0f} ms | elaborazione "
                          f"{1000 * np.mean(proc_times_log[-50:]):.1f} ms   ",
                          end="", flush=True)

                # 5. Log
                if final_error is not None:
                    errors_log.append(final_error)
                proc_times_log.append(proc_time)
                acq_times_log.append(acq_time)
                csv_writer.writerow([
                    f"{now - t0:.3f}", state,
                    "" if final_error is None else f"{final_error:.4f}",
                    f"{smoothed_error:.4f}",
                    left_pwm, right_pwm,
                    f"{acq_time:.4f}", f"{proc_time:.4f}",
                ])

                # 6. Debug
                if cv2 is not None:
                    if draw_debug(cv2, bayer_to_rgb(raw), mask, results, zones,
                                  smoothed_error, state):
                        break

                # 7. Frequenza ciclo
                remaining = 1.0 / CONTROL_HZ - (time.monotonic() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)

    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.")
    except PixyError as e:
        print(f"\nErrore Pixy: {e}")
        exit_code = 1
    finally:
        print("Arresto motori e chiusura risorse...")

        if ser is not None:
            try:
                send_motors(ser, 0, 0)
                ser.flush()
            except Exception:
                pass
            try:
                ser.close()
            except Exception:
                pass

        csv_file.close()
        executor.shutdown(wait=True)

        if cv2 is not None:
            cv2.destroyAllWindows()

        try:
            save_summary(errors_log, proc_times_log, acq_times_log)
            print(f"Statistiche salvate in {LOG_PATH_JSON}")
        except Exception as e:
            print(f"Impossibile salvare il riepilogo: {e}")

        if args.save_images > 0 or saved_images:
            extra = " (piu' altrettante _debug.png)" if SALVA_IMMAGINI_DEBUG else ""
            print(f"{saved_images} immagini salvate{extra} in {IMAGE_DIR}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())