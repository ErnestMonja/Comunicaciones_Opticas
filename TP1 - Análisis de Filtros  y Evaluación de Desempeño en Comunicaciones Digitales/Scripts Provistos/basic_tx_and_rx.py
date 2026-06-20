import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.special import erfc

# -------------------------------------------------
# Basic TX QAM
# -------------------------------------------------

enable_plots = True

M = 16
L = int(1e4) # -> TENER EN CUENTA PARA EbNo altos que sea lo suficientemente largo L
BR = 32e9
N = 2
rolloff = 0.1
h_taps = 203
EbNo_db = 10

T = 1/BR
fs = N * BR
Ts = 1/fs

# -------------------------------------------------
# QAM Modulator (equivalente a qammod)
# -------------------------------------------------

def qammod(x, M):
    m = int(np.sqrt(M))
    re = 2*(x % m) - m + 1
    im = 2*(x // m) - m + 1
    return re + 1j*im

# Símbolos
x_aux = np.random.randint(0, M, L)
ak = qammod(x_aux, M)

# -------------------------------------------------
# Upsampling
# -------------------------------------------------

xup = np.zeros(L*N, dtype=complex)
xup[::N] = ak
xup = N * xup   

# -------------------------------------------------
# Filtro RRC
# -------------------------------------------------
def round_odd(n):
    n = int(np.round(n))
    if n % 2 == 0:
        n += 1
    return n

def root_raised_cosine(fc, fs, rolloff, n_taps, t0=0):

    Ts = 1 / fs
    rolloff = rolloff + 0.0001
    T = 1 / fc

    # Force to odd
    n_taps = round_odd(n_taps)

    # Time vector
    n = np.arange(-(n_taps - 1)//2, (n_taps - 1)//2 + 1)
    t_v = n * Ts + t0
    tn_v = t_v * 2 / T

    # Filter taps 
    numerator = (
        np.sin(np.pi * (1 - rolloff) * tn_v)
        + 4 * rolloff * tn_v * np.cos(np.pi * (1 + rolloff) * tn_v)
    )

    denominator = (
        np.pi * tn_v * (1 - (4 * rolloff * tn_v)**2)
    )

    h_v = numerator / denominator

    # Valor central (evita NaN en tn_v = 0)
    center = (n_taps - 1) // 2
    h_v[center] = (1 + rolloff * (4/np.pi - 1))

    # Normalización
    h_v = h_v / np.sum(h_v)

    return h_v

h = root_raised_cosine(BR/2, fs, rolloff, h_taps, 0)

h_delay = 0  

yup = lfilter(h, 1, np.concatenate([xup, np.zeros(h_delay)]))
yup = yup[h_delay:]

# -------------------------------------------------
# Channel
# -------------------------------------------------

k = np.log2(M)
EbNo = 10**(EbNo_db/10)
SNR_slc = EbNo * k
SNR_ch = SNR_slc / N

Ps = np.var(yup)
Pn = Ps / SNR_ch

n = np.sqrt(Pn/2) * (
    np.random.randn(len(yup)) +
    1j*np.random.randn(len(yup))
)

rx = yup + n

# -------------------------------------------------
# RX
# -------------------------------------------------

h_mf = np.conj(h[::-1])
ymf = lfilter(h_mf, 1, np.concatenate([rx, np.zeros(h_delay)]))
ymf = ymf[h_delay:]

PHASE = 0
rx_down = ymf[PHASE::N] # y[k]

# -------------------------------------------------
# DELAY ESTIMATION USING CROSS-CORRELATION
# -------------------------------------------------

def estimate_delay(tx_symbols, rx_symbols):

    # usamos parte central para evitar transitorios
    Lcorr = min(len(tx_symbols), len(rx_symbols))
    tx = tx_symbols[:Lcorr]
    rx = rx_symbols[:Lcorr]

    # correlación cruzada compleja
    corr = np.correlate(rx, tx, mode='full')

    delay = np.argmax(np.abs(corr)) - (Lcorr - 1)

    return delay

# Estimar delay en símbolos
delay_est = estimate_delay(ak, rx_down)

print(f"\nEstimated symbol delay = {delay_est}")

# Compensar delay
if delay_est > 0:
    rx_aligned = rx_down[delay_est:]
    tx_aligned = ak[:len(rx_aligned)]
elif delay_est < 0:
    tx_aligned = ak[-delay_est:]
    rx_aligned = rx_down[:len(tx_aligned)]
else:
    rx_aligned = rx_down
    tx_aligned = ak[:len(rx_aligned)]

# -------------------------------------------------
# Constellation Plot
# -------------------------------------------------

if enable_plots:
    plt.figure()
    pts = rx_down[500:-100]
    plt.plot(np.real(pts), np.imag(pts), 'o')
    plt.xlabel('In phase')
    plt.ylabel('In Quadrature')
    plt.title(f'Constellation QAM-{M}, EbNo = {EbNo_db:.0f} dB')
    if M == 4:
        plt.xlim([-2,2])
        plt.ylim([-2,2])
    if M == 16:
        LIM = 5
        plt.xlim([-LIM,LIM])
        plt.ylim([-LIM,LIM])
    else:
        LIM = 9
        plt.xlim([-LIM,LIM])
        plt.ylim([-LIM,LIM])
    plt.grid(True)
    plt.show()

# -------------------------------------------------
# SLICER
# -------------------------------------------------

def slicer(rx, M):
    const = qammod(np.arange(M), M)  # todos los símbolos posibles
    
    idx = np.argmin(np.abs(rx[:, None] - const), axis=1)
    
    return const[idx]

ak_hat = slicer(rx_aligned, M)

# -------------------------------------------------
# BER
# -------------------------------------------------

def ber_theoretical(EbNo_db, M):
    k = np.log2(M)
    EbNo = 10**(EbNo_db/10)
    return (4/k)*(1-1/np.sqrt(M))*0.5*erfc(
        np.sqrt(3*k*EbNo/(M-1))/np.sqrt(2)
    )

ber_theo = ber_theoretical(EbNo_db, M)

# Solo quiero medir BER del final de mi simulación
use_frac = 0.6  # fracción de la señal que querés usar (ej: 0.6 = 60%)

start = int((1 - use_frac) * len(ak_hat))

ak_hat_cut = ak_hat[start:]
tx_cut = tx_aligned[start:]

# SER simulada
n_errors = np.sum(ak_hat_cut != tx_cut)
ser_sim = n_errors / len(ak_hat_cut)

# BER aproximada
ber_sim = ser_sim / np.log2(M)

# Some prints
print("\n - Theo BER = %.2e" % ber_theo)
print("\n - Sim SER  = %.2e" % ser_sim)
print("\n - Sim BER  = %.2e" % ber_sim)
print("\n - Errors   = %d" % n_errors)