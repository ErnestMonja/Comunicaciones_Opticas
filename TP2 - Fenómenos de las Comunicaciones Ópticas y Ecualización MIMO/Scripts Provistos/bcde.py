import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.special import erfc

# -------------------------------------------------
# Basic TX QAM
# -------------------------------------------------

enable_plots = True

M = 16
L = int(2e4)
BR = 120e9
N = 4
rolloff = 0.1
h_taps = 203
EbNo_db = 1000

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

# CD

L_fiber = 120e3         # longitud fibra [m]
D = 20e-6               # ps/(nm km) -> convertido a s/(m^2)
lambda0 = 1550e-9       # longitud de onda [m]
c = 3e8                 # velocidad de la luz

# beta2 [s^2/m]
beta2 = -(D * lambda0**2) / (2*np.pi*c)

# NTAPS
CD_MAX = round(L_fiber * D, 3) 
NTAPS_CD = round( (1.1 * fs * fs * CD_MAX * lambda0**2) / c)

fs_bcde = fs # ESTO NO ES ASI EN LA IMPLEMENTACIÓN
NTAPS_BCDE = round( (1.1 * fs * BR * CD_MAX * lambda0**2) / c)

print(f' -- CD_MAX: {CD_MAX} [ns/nm]--')
print(f' -- NTAPS CD: {NTAPS_CD } --')
print(f' -- NTAPS CD: {NTAPS_BCDE } --')

def cd_fir(beta2, L, fs, Nfft=16*2048):

    # vector de frecuencias angulares
    w = 2*np.pi * np.fft.fftfreq(Nfft, d=1/fs)

    # respuesta en frecuencia de CD
    Hcd = np.exp(1j * beta2 / 2 * L * w**2 ) # H(w)

    # respuesta temporal
    h = np.fft.ifft(Hcd) # h(t) = F^-1(H(w))

    # centrar el impulso
    h = np.fft.fftshift(h)

    # normalizar energía
    h = h / np.sqrt(np.sum(np.abs(h)**2))

    return h

h_cd = cd_fir(beta2, L_fiber, fs)

rx_w_cd = lfilter(h_cd, 1, yup)

# Noise

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

rx_noisy = rx_w_cd + n

# -------------------------------------------------
# RX
# -------------------------------------------------

bcde_enable = True

# BCD Equalizer
h_cd_zf = cd_fir(beta2, -L_fiber, fs)

if bcde_enable:
    rx_wo_cd = lfilter(h_cd_zf, 1, rx_noisy)
else:
    rx_wo_cd = rx_noisy

# MF del RRC
h_mf = np.conj(h[::-1])
ymf = lfilter(h_mf, 1, np.concatenate([rx_wo_cd, np.zeros(h_delay)]))
ymf = ymf[h_delay:]

PHASE = 2
rx_down = ymf[PHASE::N]

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

# -------------------------------------------------
# Plots
# -------------------------------------------------

def freq_response(h, fs):

    NFFT = 4*len(h)
    H = np.fft.fftshift(np.fft.fft(h, NFFT))
    f = np.fft.fftshift(np.fft.fftfreq(len(H), d=1/fs))

    return f, H

if enable_plots:
    # Constelacion
    plt.figure()
    pts = rx_down[-200:-100]
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

    # CD
    f_cd, H_cd = freq_response(h_cd, fs)
    f_eq, H_eq = freq_response(h_cd_zf, fs)

    phi_cd = -beta2*L_fiber*(2*np.pi*f_cd)**2/2
    phi_eq = +beta2*L_fiber*(2*np.pi*f_cd)**2/2

    plt.figure(figsize=(10,6))

    # fase
    plt.subplot(2,1,1)

    plt.plot(f_cd/1e9, phi_cd, label='CD theory')
    plt.plot(f_cd/1e9, phi_eq, '--', label='BCDE theory')

    plt.ylabel("Phase [rad]")
    plt.title("Phase response (parabolic)")
    plt.legend()
    plt.grid()

    # magnitud
    plt.subplot(2,1,2)

    plt.plot(f_cd/1e9, np.abs(H_cd), label='|Hcd|')
    plt.plot(f_cd/1e9, np.abs(H_eq), '--', label='|Heq|')

    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Magnitude")
    plt.title("Magnitude response ")
    plt.legend()
    plt.grid()
    plt.ylim(0,1.1)

    plt.tight_layout()
    
    plt.figure(figsize=(10,6))

    # Plot in time
    plt.subplot(2,1,1)

    plt.plot(np.real(h_cd), label='Re{h_cd}')
    plt.plot(np.imag(h_cd), '--', label='Im{h_cd}')

    plt.title("CD impulse response")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    plt.subplot(2,1,2)

    plt.plot(np.real(h_cd_zf), label='Re{h_cd_zf}')
    plt.plot(np.imag(h_cd_zf), '--', label='Im{h_cd_zf}')

    plt.title("BCDE impulse response")
    plt.xlabel("samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
