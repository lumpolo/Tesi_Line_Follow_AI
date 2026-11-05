import serial
import time

# Apri la porta seriale a 9600 baud
ser = serial.Serial('/dev/ttyACM0', 9600)  
print(f"Connesso alla porta: {ser.name}")         


# La tua variabile intera
mio_numero = -2 

# .encode() trasforma la stringa in byte pronti per essere sparati sulla seriale
dati_da_inviare = f"{mio_numero}\n".encode()

# Invia il dato
ser.write(dati_da_inviare)     
print(f"Inviato: {mio_numero}")

# Chiudi la porta
ser.close()             
