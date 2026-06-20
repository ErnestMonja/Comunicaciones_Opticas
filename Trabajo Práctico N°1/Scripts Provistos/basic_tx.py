# -----------------------------------------------------------------------------
#                                   UBA
# Programmer(s): Francisco G. Rainero
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.fft import fft, fftshift

plt.close('all')

# -------------------------------
# Basic TX M-QAM
# -------------------------------
L = 10000          # Simulation Length
BR = 32e9          # Baud Rate
N = 4              # Oversampling rate
rolloff = 0.1      # Pulse shaping rolloff
h_taps = 101       # Pulse shaping taps
M = 16              # Modulation order. QPSK (QAM4): M=4 y QAM-16: M=16

fs = N * BR        # Sampling rate
T = 1 / BR         # Symbol period
Ts = 1 / fs        # Sample period

# -------------------------------
# QAM MODULATION (Gray mapping)
# -------------------------------
# Genera una secuencia aleatoria de símbolos enteros entre 0 y M-1.
# L es la cantidad de símbolos generados.
x_aux = np.random.randint(0, M, L)

# Función que realiza la modulación M-QAM.
# Recibe un vector de índices de símbolos (0 ... M-1) y los mapea a la
# constelación QAM correspondiente.
def qammod(symbols, M):
    # m es la cantidad de niveles por eje (I y Q).
    # Para QAM cuadrada se cumple que M = m^2 (por ejemplo: 16-QAM → m=4).
    m = int(np.sqrt(M))
    
    # Calcula la componente real (I) del símbolo:
    # symbols % m da la posición horizontal dentro de la grilla.
    # La expresión genera niveles igualmente espaciados: {-m+1, ..., m-1}.
    re = 2*(symbols % m) - m + 1
    
    # Calcula la componente imaginaria (Q):
    # symbols // m determina la fila vertical dentro de la grilla.
    im = 2*(symbols // m) - m + 1
    
    # Forma el símbolo complejo I + jQ .
    return (re + 1j*im) 

# Convierte los símbolos enteros aleatorios en símbolos complejos M-QAM
# que luego pueden transmitirse en el sistema de comunicaciones.
ak = qammod(x_aux, M)

# -------------------------------
# Upsampling
# -------------------------------
xup = np.zeros(L*N, dtype=complex)
xup[::N] = ak * N

# -------------------------------
# Raised Cosine Filter
# -------------------------------
def raised_cosine(beta, span, sps):
    t = np.arange(-span//2, span//2 + 1) / sps
    h = np.zeros_like(t)

    for i in range(len(t)):
        if t[i] == 0.0:
            h[i] = 1.0
        elif beta != 0 and abs(t[i]) == 1/(2*beta):
            h[i] = (np.pi/4)*np.sinc(1/(2*beta))
        else:
            h[i] = (np.sinc(t[i]) *
                    np.cos(np.pi*beta*t[i]) /
                    (1 - (2*beta*t[i])**2))
    return h/np.sum(h)

h = raised_cosine(rolloff, h_taps, N)
h_delay = (h_taps - 1) // 2

yup = signal.lfilter(h, 1, np.concatenate([xup, np.zeros(h_delay)]))
yup = yup[h_delay+1:]

# -------------------------------
# Time domain plot
# -------------------------------
# Tiempo asociado a símbolos y muestras
t_symbols = np.arange(L) * T
t_samples = np.arange(len(yup)) * Ts

# Cantidad de muestras que vamos a mostrar
LPLOT = 30 * N

plt.figure(figsize=(6,6))

plt.subplot(2,1,1)

# Señal filtrada (parte real)
plt.plot(t_samples[:LPLOT], np.real(yup[:LPLOT]), '-x', label="Filtered signal")

# Símbolos originales
plt.stem(t_symbols[:30],
         np.real(ak[:30]),
         linefmt='r-',
         markerfmt='ro',
         basefmt=" ",
         label="Tx symbols")

plt.title("TX Real Part")
plt.legend()
plt.grid(True)

plt.subplot(2,1,2)

plt.plot(t_samples[:LPLOT], np.imag(yup[:LPLOT]), '-x', label="Filtered signal")

plt.stem(t_symbols[:30],
         np.imag(ak[:30]),
         linefmt='r-',
         markerfmt='ro',
         basefmt=" ",
         label="Tx symbols")

plt.title("TX Imag Part")
plt.legend()
plt.grid(True)

plt.tight_layout()

# -------------------------------
# Filter Frequency Response
# -------------------------------
NFFT = 1024*8
H_abs = np.abs(fft(h, NFFT))
f = np.linspace(-fs/2, fs/2, NFFT)

plt.figure(figsize=(8,5))
plt.plot(f/1e9, fftshift(20*np.log10(H_abs/np.max(H_abs))))
plt.title("Filter Frequency Response (dB)")
plt.xlabel("Frequency [GHz]")
plt.grid(True)

plt.figure(figsize=(8,5))
plt.plot(h)
plt.title("Filter Frequency Response")
plt.xlabel("Samples")
plt.grid(True)

# -------------------------------
# PSD using Welch
# -------------------------------
plt.figure(figsize=(10,5))

f1, Pxx1 = signal.welch(ak, BR, nperseg=NFFT//2, return_onesided=False)
f3, Pxx3 = signal.welch(xup, fs, nperseg=NFFT//4, return_onesided=False)
f2, Pxx2 = signal.welch(yup, fs, nperseg=NFFT//4, return_onesided=False)

plt.plot(fftshift(f1)/1e9,
         fftshift(10*np.log10(Pxx1/np.max(Pxx1))),
         label="PSD symbols")

plt.plot(fftshift(f2)/1e9,
         fftshift(10*np.log10(Pxx2/np.max(Pxx2))),
         label="PSD filtered")

plt.plot(fftshift(f3)/1e9,
         fftshift(10*np.log10(Pxx3/np.max(Pxx3))),
         label="PSD xup")

plt.legend()
plt.xlabel("Frequency [GHz]")
plt.ylabel("PSD [dB]")
plt.grid(True)

# -------------------------------
# Eye Diagram
# -------------------------------
def eye_diagram(signal, sps, num_symbols=200):
    samples = sps*2
    truncated = signal[:num_symbols*samples]
    reshaped = truncated.reshape((-1, samples))
    plt.figure()
    for row in reshaped:
        plt.plot(np.real(row), alpha=0.3)
    plt.title("Eye Diagram")
    plt.grid(True)

eye_diagram(yup[500:5000], N)

# -------------------------------
# Constellation Diagram
# -------------------------------
# Muestra los símbolos transmitidos en el plano complejo
# (I vs Q). Esto permite verificar la constelación QAM.

plt.figure(figsize=(6,6))

plt.scatter(np.real(ak), np.imag(ak),
            s=5,
            alpha=0.5)

plt.title("TX Constellation (16-QAM)")
plt.xlabel("In-Phase (I)")
plt.ylabel("Quadrature (Q)")
plt.grid(True)
plt.axis("equal")


plt.show()