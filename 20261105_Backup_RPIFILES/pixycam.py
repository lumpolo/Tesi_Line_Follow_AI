"""
pixycam.py - acquisizione frame dalla Pixy / Pixy2 (CMUcam5) in Python.

Uso:
    from pixycam import PixyCam

    with PixyCam() as cam:
        img = cam.grab()        # numpy uint8 (altezza, larghezza, 3), canali RGB
        raw = cam.grab_raw()    # numpy uint8 (altezza, larghezza), Bayer grezzo

Dimensioni: Pixy2 316x208, Pixy 320x200.

Con OpenCV:  bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

Serve libpixyframe.so (creata da install_pixy.sh) nella stessa cartella
di questo file, oppure il suo percorso nella variabile PIXY_LIB.
Apri una sola PixyCam per programma.
"""
import ctypes
import os

import numpy as np

MAX_PIXELS = 1280 * 800

_LIB_PATH = os.environ.get(
    "PIXY_LIB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "libpixyframe.so"),
)

_ERRORS = {
    -1: "errore di I/O sulla USB",
    -4: "la Pixy si e' scollegata",
    -5: "Pixy non trovata. E' collegata? Se si', mancano i permessi: "
        "rilancia install_pixy.sh e ricollega la Pixy, oppure usa sudo",
    -6: "USB occupata: un altro programma (es. PixyMon) sta usando la Pixy",
    -3: "risposta della Pixy non riconosciuta: aggiornate install_pixy.sh e rilanciatelo",
    -7: "la Pixy non risponde (timeout USB)",
    -101: "la Pixy non risponde (timeout)",
    -102: "la Pixy non risponde (timeout)",
    -151: "errore di comunicazione con la Pixy",
    -152: "comando non riconosciuto dal firmware della Pixy",
    -200: "la Pixy ha rifiutato la richiesta del frame (copiate l'output con PIXY_DEBUG=1)",
    -201: "risposta della Pixy incoerente",
    -202: "frame piu' grande del previsto",
    -203: "la Pixy ha risposto senza immagine (copiate l'output con PIXY_DEBUG=1)",
}


class PixyError(RuntimeError):
    pass


def _error(code):
    return PixyError(f"{_ERRORS.get(code, 'errore sconosciuto')} (codice {code})")


def bayer_to_rgb(raw):
    """Converte il frame Bayer della Pixy (schema BGGR) in RGB.

    Interpolazione bilineare, la stessa che usa PixyMon.
    """
    p = np.pad(raw, 1, mode="reflect").astype(np.uint16)
    c = p[1:-1, 1:-1]
    up, down = p[:-2, 1:-1], p[2:, 1:-1]
    left, right = p[1:-1, :-2], p[1:-1, 2:]
    cross = (up + down + left + right) >> 2
    diag = (p[:-2, :-2] + p[:-2, 2:] + p[2:, :-2] + p[2:, 2:]) >> 2
    horiz = (left + right) >> 1
    vert = (up + down) >> 1

    h, w = raw.shape
    odd_y = (np.arange(h) & 1).astype(bool)[:, None]
    odd_x = (np.arange(w) & 1).astype(bool)[None, :]
    is_r = odd_y & odd_x            # righe dispari, colonne dispari
    is_b = ~odd_y & ~odd_x          # righe pari, colonne pari
    g_in_r_row = odd_y & ~odd_x     # verde sulla riga dei rossi
    g_in_b_row = ~odd_y & odd_x     # verde sulla riga dei blu

    r = np.select([is_r, g_in_r_row, g_in_b_row], [c, horiz, vert], diag)
    g = np.where(is_r | is_b, cross, c)
    b = np.select([is_b, g_in_b_row, g_in_r_row], [c, horiz, vert], diag)
    return np.dstack([r, g, b]).astype(np.uint8)


class PixyCam:
    def __init__(self, lib_path=_LIB_PATH):
        if not os.path.exists(lib_path):
            raise PixyError(f"{lib_path} non trovato: lancia prima install_pixy.sh")
        lib = ctypes.CDLL(lib_path)
        lib.pf_init.restype = ctypes.c_int
        lib.pf_get_frame.restype = ctypes.c_int
        lib.pf_get_frame.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
        ]
        lib.pf_close.restype = None
        self._lib = lib
        self._buf = np.empty(MAX_PIXELS, dtype=np.uint8)
        self._w = ctypes.c_uint16()
        self._h = ctypes.c_uint16()

        rv = lib.pf_init()
        if rv < 0:
            raise _error(rv)
        self._open = True

    def grab_raw(self):
        """Un frame grezzo Bayer: numpy uint8 (altezza, larghezza)."""
        n = self._lib.pf_get_frame(
            self._buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            self._buf.size,
            ctypes.byref(self._w),
            ctypes.byref(self._h),
        )
        if n < 0:
            raise _error(n)
        return self._buf[:n].reshape(self._h.value, self._w.value).copy()

    def grab(self):
        """Un frame a colori: numpy uint8 (altezza, larghezza, 3) RGB."""
        return bayer_to_rgb(self.grab_raw())

    def close(self):
        """Rimette in funzione il programma a bordo della Pixy e chiude la USB."""
        if getattr(self, "_open", False):
            self._lib.pf_close()
            self._open = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()
