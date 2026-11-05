#!/usr/bin/env python3
"""
Acquisisce frame dalla Pixy e li salva come PNG.

    python3 acquisisci.py              # 10 frame in ./frames
    python3 acquisisci.py -n 100       # 100 frame
    python3 acquisisci.py -n 0         # continua finche' non premi Ctrl+C
    python3 acquisisci.py --raw        # salva anche il Bayer grezzo (.npy)
    python3 acquisisci.py --solo-raw   # salva solo il Bayer grezzo (piu' veloce)

Alla fine stampa quanto tempo va, in media, in ogni fase.
"""
import argparse
import os
import time

import numpy as np
from PIL import Image

from pixycam import PixyCam, bayer_to_rgb

def acquisisci_frames(numero=0, con_rgb=True):
    """
    Generatore che restituisce una coppia (rgb, raw) per ogni frame.
 
    rgb:  numpy uint8 (altezza, larghezza, 3), canali RGB.
    raw:  numpy uint8 (altezza, larghezza), Bayer grezzo.
 
    numero: frame da acquisire, 0 = continuo.
    con_rgb: se False salta la demosaicizzazione (rgb = None): molto piu' veloce,
             utile quando basta il frame grezzo.
 
    La Pixy viene aperta al primo next() e chiusa quando il generatore
    termina o viene chiuso con close() / contextlib.closing().
    Eventuali errori di comunicazione sollevano PixyError.
    """
    if numero < 0:
        raise ValueError("Il numero di frame non può essere negativo.")
 
    with PixyCam() as cam:
        i = 0
 
        while numero == 0 or i < numero:
            raw = cam.grab_raw()
            rgb = bayer_to_rgb(raw) if con_rgb else None
 
            i += 1
            yield rgb, raw

            
def main():
    ap = argparse.ArgumentParser(description="Acquisizione frame dalla Pixy CMUcam5")
    ap.add_argument("-n", type=int, default=10, help="numero di frame (0 = infiniti)")
    ap.add_argument("-o", default="frames", help="cartella di uscita")
    ap.add_argument("--raw", action="store_true", help="salva anche il frame grezzo .npy")
    ap.add_argument("--solo-raw", action="store_true",
                    help="salva solo il frame grezzo .npy (niente conversione ne' PNG)")
    args = ap.parse_args()

    os.makedirs(args.o, exist_ok=True)
    t_grab = t_conv = t_save = 0.0

    with PixyCam() as cam:
        print("Pixy collegata. Ctrl+C per fermare.")
        i = 0
        t0 = time.monotonic()
        try:
            while args.n == 0 or i < args.n:
                ta = time.monotonic()
                raw = cam.grab_raw()
                tb = time.monotonic()
                name = os.path.join(args.o, f"frame_{i:05d}")
                if args.solo_raw:
                    tc = tb
                    np.save(name + ".npy", raw)
                else:
                    img = bayer_to_rgb(raw)
                    tc = time.monotonic()
                    Image.fromarray(img).save(name + ".png", compress_level=1)
                    if args.raw:
                        np.save(name + ".npy", raw)
                td = time.monotonic()
                t_grab += tb - ta
                t_conv += tc - tb
                t_save += td - tc
                i += 1
                fps = i / (td - t0)
                print(f"\r{i} frame  ({fps:.1f} fps)", end="", flush=True)
        except KeyboardInterrupt:
            pass
    print(f"\nSalvati {i} frame in {os.path.abspath(args.o)}")
    if i:
        ms = lambda t: f"{t / i * 1000:.0f} ms"
        print(f"Tempo medio per frame: acquisizione {ms(t_grab)}, "
              f"conversione colori {ms(t_conv)}, salvataggio {ms(t_save)}")


if __name__ == "__main__":
    main()
