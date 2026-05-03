# -----------------------------------------------------------------------------
#                                   MARVELL
# Programmer(s): Francisco G. Rainero
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.fft import fft, fftshift
from scipy.special import erfc

plt.close('all')

# -------------------------------
# Basic TX M-QAM
# -------------------------------
L = 10000          # Simulation Length
BR = 32e9          # Baud Rate
M = 4              # Modulation order. QPSK (QAM4): M=4 y QAM-16: M=16
k = int(np.log2(M))     # k = LOG2(M)

fs = BR            # Sampling rate
T = 1 / BR         # Symbol period
Ts = 1 / fs        # Sample period

snr_db = 7 

# Primero vamos es generar los bits
bits = np.random.randint(0,2, k*L)

# Los voy a agrupar cada k
bits_reshaped = bits.reshape((L,k))

# Pasar al mapeo de la señal
def map_bits_to_qpsk(b1, b0):
    if (b1, b0) == (0, 0):
        return complex(1,1)
    if (b1, b0) == (0, 1):
        return complex(1,-1)
    if (b1, b0) == (1, 0):
        return complex(-1,1)
    if (b1, b0) == (1, 1):
        return complex(-1,-1)

if M == 4:
    symbols = [map_bits_to_qpsk(b[0], b[1]) for b in bits_reshaped]
    
symbols = np.array(symbols)

# Agregado de ruido
Ps = np.var(symbols)
snr = 10**(snr_db/10)
Pn = Ps/snr

# Option 1
# noise = np.sqrt(Pn/2) * ( np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols)) )

# Option 2
noise_real  = np.sqrt(Pn/2) *  np.random.randn(len(symbols))
noise_imag  = np.sqrt(Pn/2) *  np.random.randn(len(symbols))
noise = noise_real + 1j * noise_imag

symbols_noisy = symbols + noise

# Slicer
def qpsk_slicer(rx):
    I_hat = np.sign(rx.real)
    Q_hat = np.sign(rx.imag)
    
    return I_hat + 1j*Q_hat

symbols_hat = qpsk_slicer(symbols_noisy)

# Medir SER
errores = np.sum(symbols_hat != symbols)
SER = errores / len(symbols)
BER_aprox = SER / k

print(" - Cantidad de errores: ")
print(errores) # Para buena estadística, debemos contar al menos 100 errores
print(" - SER: ")
print(SER)
print(" - BER simulada: ")
print(BER_aprox) # Deberia parecerse a la BER "teorica"

# BER Toerica de QPSK
def Q(x):
    return 0.5 * erfc(x / np.sqrt(2))

# BER teórica QPSK
ber_theo = Q(np.sqrt(snr))

print(" - BER teórica: ")
print(ber_theo) 

# Plot de un histograma 
#plt.figure()
#plt.hist(symbols_noisy.real, bins=50)
#plt.title(f"QPSK con AWGN (Real) - SNR = {snr_db} dB")
#plt.xlabel("Valor")
#plt.ylabel("Frecuencia")
#plt.grid()

#plt.figure()
#lt.hist(symbols_noisy.imag, bins=50)
#plt.title(f"QPSK con AWGN (Imag) - SNR = {snr_db} dB")
#lt.xlabel("Valor")
#plt.ylabel("Frecuencia")
#plt.grid()
#plt.show()

