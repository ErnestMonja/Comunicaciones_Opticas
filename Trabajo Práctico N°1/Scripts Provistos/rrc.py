import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift

plt.close("all")

# -------------------------------------------------
# Parámetros 
# -------------------------------------------------

BR = 32e9
N = 4
rolloff = 0.1
h_taps = 101

fs = N * BR
Ts = 1 / fs

# -------------------------------------------------
# Función round_odd 
# -------------------------------------------------

def round_odd(n):
    n = int(np.round(n))
    if n % 2 == 0:
        n += 1
    return n

# -------------------------------------------------
# Raised Cosine 
# -------------------------------------------------

def raised_cosine(fc, fs, rolloff, n_taps, t0=0):

    rolloff = rolloff + 0.0001
    Ts = 1 / fs
    T = 1 / fc

    n_taps = round_odd(n_taps)

    n = np.arange(-(n_taps - 1)//2, (n_taps - 1)//2 + 1)
    t_v = n * Ts + t0
    tn_v = t_v * 2 / T

    # np.sinc ya es sinc normalizado 
    h_v = np.sinc(tn_v) * np.cos(np.pi * rolloff * tn_v) \
          / (1 - (2 * rolloff * tn_v)**2)

    h_v = h_v / np.sum(h_v)

    return h_v

# -------------------------------------------------
# Root Raised Cosine = sqrt(raised cosine en freq)
# -------------------------------------------------

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

# -------------------------------------------------
# Generación filtros 
# -------------------------------------------------

h_rrc = root_raised_cosine(BR/2, fs, rolloff, h_taps)

h_rrc_rrc = np.convolve(h_rrc, h_rrc)

h_rc = raised_cosine(BR/2, fs, rolloff, len(h_rrc_rrc))

# -------------------------------------------------
# FFTs 
# -------------------------------------------------

NFFT = 2048
f = np.arange(-NFFT/2, NFFT/2) * fs / NFFT

H_RRC = fftshift(np.abs(fft(h_rrc, NFFT)))
H_RRC_RRC = H_RRC * H_RRC
H_RC = fftshift(np.abs(fft(h_rc, NFFT)))

# -------------------------------------------------
# PLOTS (idénticos)
# -------------------------------------------------

n_rrc_v = np.arange(-(h_taps-1)//2, (h_taps-1)//2 + 1)
n_rc_v = np.arange(-(len(h_rrc_rrc)-1)//2, (len(h_rrc_rrc)-1)//2 + 1)

plt.figure(figsize=(5,5))
plt.plot(n_rrc_v, h_rrc, '--k', linewidth=1)
plt.plot(n_rc_v, h_rrc_rrc, 'r', linewidth=2)
plt.plot(n_rc_v, h_rc, '--b', linewidth=1.5)
plt.title('hrrc*hrrc = hrc')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend(['hrrc','hrrc*hrrc','hrc'])

plt.figure(figsize=(5,5))
plt.plot(f/1e9, H_RRC, '--k', linewidth=1)
plt.plot(f/1e9, H_RRC_RRC, 'r', linewidth=2)
plt.plot(f/1e9, H_RC, '--b', linewidth=1.5)
plt.axvline(BR/2/1e9, linestyle='--', color='k')
plt.title('Hrrc.Hrrc = Hrc')
plt.xlabel('Freq [GHz]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend(['Hrrc','Hrrc.Hrrc','Hrc'])

plt.figure(figsize=(5,5))
plt.plot(f/1e9, 20*np.log10(H_RRC), '--k', linewidth=1)
plt.plot(f/1e9, 20*np.log10(H_RRC_RRC), 'r', linewidth=2)
plt.plot(f/1e9, 20*np.log10(H_RC), '--b', linewidth=1.5)
plt.axvline(BR/2/1e9, linestyle='--', color='k')
plt.title('Hrrc.Hrrc = Hrc')
plt.xlabel('Freq [GHz]')
plt.ylabel('Amplitude [dB]')
plt.grid(True)
plt.legend(['Hrrc','Hrrc.Hrrc','Hrc'])

plt.show()