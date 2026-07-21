import numpy as np
import scipy.signal as sig
from scipy.special import erfc

# Filtro Coseno Realzado
def root_raised_cosine(NOS, r, delay):
    """Filtro de Raiz Coseno Realzado (RRC)."""
    N = 2 * delay * NOS + 1
    t = np.arange(N) - N // 2
    t = t / NOS
    h = np.zeros(N)
    for i, val in enumerate(t):
        if val == 0.0:
            h[i] = 1.0 - r + (4 * r / np.pi)
        elif r != 0 and abs(val) == 1.0 / (4 * r):
            h[i] = (r / np.sqrt(2)) * (((1 + 2 / np.pi) * np.sin(np.pi / (4 * r))) + 
                                       ((1 - 2 / np.pi) * np.cos(np.pi / (4 * r))))
        else:
            h[i] = (np.sin(np.pi * val * (1 - r)) + 4 * r * val * np.cos(np.pi * val * (1 + r))) / \
                   (np.pi * val * (1 - (4 * r * val) ** 2))
    return h / np.sqrt(np.sum(h**2))

# Modulador QAM
def qammod(symbols, M):
    m = int(np.sqrt(M))
    niv = symbols % m
    re = 2 * niv - m + 1
    niv = symbols // m
    im = 2 * niv - m + 1
    return re + 1j * im

# Slicer
def slicer_QAM(syms, M):
    m = int(np.sqrt(M))
    real_part = np.clip(2 * np.round((np.real(syms) - 1) / 2) + 1, -m + 1, m - 1)
    imag_part = np.clip(2 * np.round((np.imag(syms) - 1) / 2) + 1, -m + 1, m - 1)
    return real_part + 1j * imag_part

# Encuentra el retardo de y respecto a x
def finddelay(x, y):
    lags = sig.correlation_lags(len(y), len(x))
    corr = sig.correlate(y, x, mode='full')
    return lags[np.argmax(np.abs(corr))]

# Funcion Q
def Q(x):
    return 0.5 * erfc(x / np.sqrt(2))

# BER teórica para QAM-16
def ber_theoretical(snr_db):    
    snr = 10**(snr_db/10)
    return 3/4 * Q(np.sqrt(snr/5))