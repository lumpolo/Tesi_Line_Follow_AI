// Funzioni C chiamate da pixycam.py (via ctypes) per prendere i frame
// grezzi dalla Pixy (CMUcam5) o dalla Pixy2 con libpixyusb.
//
// Procedura (come la libreria ufficiale della Pixy2): "stop" del programma
// a bordo, attesa che sia fermo, poi "cam_getFrame" modalita' 0x21.
// Le risposte vengono lette come liste generiche di argomenti, cosi'
// funziona con firmware diversi (Pixy 2.0.x e Pixy2 3.0.x).
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <chrono>
#include "libusb.h"
#include <boost/thread.hpp>
#include <boost/chrono.hpp>
// serve accedere al canale Chirp e al suo mutex dentro libpixyusb
#define private public
#define protected public
#include "pixyinterpreter.hpp"
#undef private
#undef protected
#include "pixy.h"

extern PixyInterpreter interpreter;   // definito in pixy.cpp

#define PF_MODE 0x21                  // stessa modalita' usata da PixyMon

// dimensioni del frame grezzo: prima Pixy2 (316x208), poi Pixy (320x200)
static const uint16_t SIZES[2][2] = {{316, 208}, {320, 200}};
static uint16_t g_w = 0, g_h = 0;
static ChirpProc g_get_frame = -1;

static bool debug_on() { return getenv("PIXY_DEBUG") != NULL; }

// Trova pixel, larghezza e altezza in una lista di argomenti Chirp
// [risultato] ... [larghezza] [altezza] [n. pixel] [pixel].
extern "C" int pf_parse_reply(void **args, uint8_t *out, uint32_t out_size,
                              uint16_t *out_w, uint16_t *out_h)
{
    int n = 0;
    while (n < CRP_MAX_ARGS + 1 && args[n] != NULL)
        n++;
    if (n < 1)
        return -201;
    int32_t response = *(int32_t *)args[0];
    if (debug_on())
        fprintf(stderr, "pixydebug: cam_getFrame -> risultato %d, %d argomenti\n", response, n);
    if (response < 0)
        return -200;                                   // la Pixy ha rifiutato
    if (n < 3)
        return -203;                                   // risposta senza immagine

    uint32_t num_pixels = *(uint32_t *)args[n - 2];
    uint8_t *pixels     = (uint8_t *)args[n - 1];
    uint16_t dims[2] = {0, 0};
    int found = 0;
    for (int i = n - 3; i >= 1 && found < 2; i--) {
        uint8_t t = Chirp::getType(args[i]);
        if (!(t & CRP_ARRAY) && (t & 0x0f) == 2)
            dims[found++] = *(uint16_t *)args[i];
    }
    uint16_t w = found == 2 ? dims[1] : 0;
    uint16_t h = found == 2 ? dims[0] : 0;
    if ((uint32_t)w * h != num_pixels) {               // ripiego: dimensioni richieste
        w = *out_w;
        h = *out_h;
    }
    if (pixels == NULL || num_pixels == 0 || num_pixels != (uint32_t)w * h)
        return -201;                                   // risposta incoerente
    if (num_pixels > out_size)
        return -202;                                   // buffer troppo piccolo
    memcpy(out, pixels, num_pixels);
    *out_w = w;
    *out_h = h;
    return (int)num_pixels;
}

// Comando senza parametri; restituisce il valore di ritorno della Pixy.
static int simple_command(const char *name, int32_t *result)
{
    std::lock_guard<std::mutex> lock(interpreter.chirp_access_mutex_);
    ChirpProc proc = interpreter.receiver_->getProc(name);
    if (proc < 0)
        return PIXY_ERROR_INVALID_COMMAND;
    void *args[CRP_MAX_ARGS + 1];
    memset(args, 0, sizeof(args));
    int rv = interpreter.receiver_->call(SYNC_RETURN_ARRAY, proc, END_OUT_ARGS, args, END_IN_ARGS);
    if (rv < 0)
        return rv;
    *result = args[0] ? *(int32_t *)args[0] : -1;
    return 0;
}

// Una richiesta di frame w x h.
static int grab_once(uint16_t w, uint16_t h, uint8_t *out, uint32_t out_size,
                     uint16_t *out_w, uint16_t *out_h)
{
    std::lock_guard<std::mutex> lock(interpreter.chirp_access_mutex_);
    if (g_get_frame < 0) {
        g_get_frame = interpreter.receiver_->getProc("cam_getFrame");
        if (g_get_frame < 0)
            return PIXY_ERROR_INVALID_COMMAND;
    }
    void *args[CRP_MAX_ARGS + 1];
    memset(args, 0, sizeof(args));
    int rv = interpreter.receiver_->call(SYNC_RETURN_ARRAY, g_get_frame,
                                         UINT8(PF_MODE),
                                         UINT16(0), UINT16(0),     // offset x, y
                                         UINT16(w), UINT16(h),
                                         END_OUT_ARGS,
                                         args,
                                         END_IN_ARGS);
    if (rv < 0)
        return rv;
    // i dati stanno nel buffer di Chirp: vanno copiati finche' teniamo il lock
    *out_w = w;
    *out_h = h;
    return pf_parse_reply(args, out, out_size, out_w, out_h);
}

extern "C" {

// Apre la Pixy e ferma il programma a bordo (altrimenti la telecamera e'
// occupata a cercare oggetti).
int pf_init(void)
{
    int rv = pixy_init();
    if (rv < 0)
        return rv;
    if (interpreter.receiver_ == NULL)
        return PIXY_ERROR_USB_NO_DEVICE;

    // libpixyusb ha un thread che ogni 15 ms ascolta la USB tenendo occupato
    // il canale fino a 50 ms per volta: per i frame non serve e ci rallenta.
    if (interpreter.thread_.joinable()) {
        interpreter.thread_die_ = true;
        interpreter.thread_.join();
    }

    if (debug_on()) {
        uint16_t ma = 0, mi = 0, bu = 0;
        if (pixy_get_firmware_version(&ma, &mi, &bu) == 0)
            fprintf(stderr, "pixydebug: firmware %u.%u.%u (%s)\n", ma, mi, bu,
                    ma >= 3 ? "Pixy2" : "Pixy");
    }

    int32_t res = 0;
    rv = simple_command("stop", &res);
    if (rv < 0)
        return rv;
    // "stop" non e' immediato: aspettiamo che il programma sia fermo (max 2 s)
    for (int i = 0; i < 200; i++) {
        int32_t running = 1;
        if (simple_command("running", &running) < 0 || running == 0)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return 0;
}

// Copia un frame Bayer (1 byte per pixel) in 'out'.
// Ritorna il numero di byte copiati, oppure un codice di errore negativo.
int pf_get_frame(uint8_t *out, uint32_t out_size, uint16_t *out_w, uint16_t *out_h)
{
    if (interpreter.receiver_ == NULL)
        return PIXY_ERROR_USB_NO_DEVICE;

    if (g_w == 0) {
        // prima volta: scopriamo quale dimensione accetta questa telecamera
        int rv = -200;
        for (int s = 0; s < 2; s++) {
            for (int attempt = 0; attempt < 100; attempt++) {
                rv = grab_once(SIZES[s][0], SIZES[s][1], out, out_size, out_w, out_h);
                if (rv != -203)
                    break;
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            if (rv > 0) {
                g_w = SIZES[s][0];
                g_h = SIZES[s][1];
                if (debug_on())
                    fprintf(stderr, "pixydebug: frame %ux%u\n", g_w, g_h);
                return rv;
            }
            if (rv != -200)
                return rv;
        }
        return rv;
    }

    int rv = -203;
    for (int attempt = 0; attempt < 200; attempt++) {
        rv = grab_once(g_w, g_h, out, out_size, out_w, out_h);
        if (rv != -200 && rv != -203)
            return rv;
        if (debug_on())
            fprintf(stderr, "pixydebug: frame non pronto (%d), riprovo\n", rv);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return rv;
}

// Riavvia il programma a bordo e chiude la connessione.
void pf_close(void)
{
    if (interpreter.receiver_) {
        int32_t res;
        simple_command("run", &res);
    }
    pixy_close();
}

} // extern "C"
