import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift

# -------------------------------------------------
# Parámetros 
# -------------------------------------------------

BR = 32e9
N = 4
rolloff_v = np.array([0, 0.2, 0.5, 1])
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

def raised_cosine(fc, fs, rolloff_v, n_taps, t0=0):

    rolloff_v += 0.0001
    Ts = 1 / fs
    T = 1 / (2*fc) # Periodo de simbolo = 1/BR

    n_taps = round_odd(n_taps)

    n = np.arange(-(n_taps - 1)//2, (n_taps - 1)//2 + 1)
    t_v = n * Ts + t0
    tn_v = t_v / T # Normalizar a periodos de simbolo

    h_m = [0]*len(rolloff_v)

    for i in range(len(rolloff_v)):
        # np.sinc ya es sinc normalizado 
        h_m[i] = np.sinc(tn_v) * np.cos(np.pi * rolloff_v[i] * tn_v) \
            / (1 - (2 * rolloff_v[i] * tn_v)**2)

        # Normalización
        h_m[i] = h_m[i] / np.sum(h_m[i])

    return h_m

# -------------------------------------------------
# Root Raised Cosine = sqrt(raised cosine en freq)
# -------------------------------------------------

def root_raised_cosine(fc, fs, rolloff_v, n_taps, t0=0):

    rolloff_v += 0.0001
    Ts = 1 / fs
    T = 1 / (2*fc) # Periodo de simbolo = 1/BR

    # Force to odd
    n_taps = round_odd(n_taps)

    # Time vector
    n = np.arange(-(n_taps - 1)//2, (n_taps - 1)//2 + 1)
    t_v = n * Ts + t0
    tn_v = t_v / T # Normalizar a periodos de simbolo

    h_m = [0]*len(rolloff_v)

    for i in range(len(rolloff_v)):
        # Filter taps 
        numerator = (
            np.sin(np.pi * (1 - rolloff_v[i]) * tn_v)
            + 4 * rolloff_v[i] * tn_v * np.cos(np.pi * (1 + rolloff_v[i]) * tn_v)
        )

        denominator = (
            np.pi * tn_v * (1 - (4 * rolloff_v[i] * tn_v)**2)
        )

        h_m[i] = numerator / denominator

        # Valor central (evita NaN en tn_v = 0)
        center = (n_taps - 1) // 2
        h_m[i][center] = (1 + rolloff_v[i] * (4/np.pi - 1))

        # Normalización
        h_m[i] = h_m[i] / np.sum(h_m[i])

    return h_m

# -------------------------------------------------
# Generación filtros 
# -------------------------------------------------

h_rrc = root_raised_cosine(BR/2, fs, rolloff_v, h_taps)

h_rrc_rrc = [np.convolve(h_rrc[i], h_rrc[i]) for i in range(len(rolloff_v))]

h_rc = raised_cosine(BR/2, fs, rolloff_v, len(h_rrc_rrc[0]))

# -------------------------------------------------
# FFTs 
# -------------------------------------------------

NFFT = 2048
f = np.arange(-NFFT/2, NFFT/2) * fs / NFFT

H_RRC = [0]*len(rolloff_v)
H_RRC_RRC = [0]*len(rolloff_v)
H_RC = [0]*len(rolloff_v)

for i in range(len(rolloff_v)):
    H_RRC[i] = fftshift(np.abs(fft(h_rrc[i], NFFT)))
    H_RRC_RRC[i] = H_RRC[i] * H_RRC[i]
    H_RC[i] = fftshift(np.abs(fft(h_rc[i], NFFT)))

# -------------------------------------------------
# PLOTS
# -------------------------------------------------

n_rrc_v = np.arange(-(h_taps-1)//2, (h_taps-1)//2 + 1) / N
n_rc_v = np.arange(-(len(h_rrc_rrc[0])-1)//2, (len(h_rrc_rrc[0])-1)//2 + 1) / N

# Plot de varias respuestas al impulso
plt.figure(figsize=(8,5))

for i in range(len(rolloff_v)):
    plt.plot(n_rrc_v, h_rrc[i], linewidth=1.5, label=f'{np.round(rolloff_v[i], 2)}')

plt.title('Root-Raised Cosine Impulse Response')
plt.xlabel(r'Time ($\times T$)')
plt.ylabel('Amplitude')
plt.legend(title='Rolloff')
plt.grid(True)
plt.xlim([-7.5,7.5])

# Magnitud de la respuesta en frecuencia de todas las respuestas al impulso

plt.figure(figsize=(8,5))

for i in range(len(rolloff_v)):
    plt.plot(f/1e9, H_RRC[i], linewidth=1.5, label=f'{np.round(rolloff_v[i], 2)}')

plt.axvline(BR/2/1e9, linestyle='--', color='k')
plt.title('Root-Raised Cosine Frequency Response')
plt.xlabel('Frequency [GHz]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend(title='Rolloff')

plt.figure(figsize=(5,5))

for i in range(len(rolloff_v)):
    plt.plot(f/1e9, 20*np.log10(H_RRC[i]), linewidth=1.5, label=f'{np.round(rolloff_v[i], 2)}')

plt.axvline(BR/2/1e9, linestyle='--', color='k')
plt.title('Root-Raised Cosine Frequency Response in dB')
plt.xlabel('Frequency [GHz]')
plt.ylabel('Amplitude [dB]')
plt.grid(True)
plt.legend(title='Rolloff')

# Convolucion de dos filtros RRC

plt.figure(figsize=(8,5))
plt.plot(n_rrc_v, h_rrc[2], 'r', linewidth=1)
plt.plot(n_rc_v, h_rrc_rrc[2], '--k', linewidth=2)
plt.plot(n_rc_v, h_rc[2], 'b', linewidth=1)
plt.title(r'RC and RRC Impulse Response. $\beta = 0.5$')
plt.xlabel(r'Time ($\times T$)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend([r'$h_{RRC}$',r'$h_{RRC}\ast h_{RRC}$',r'$h_{RC}$'])
plt.xlim([-4,4])

plt.figure(figsize=(5,5))
plt.plot(f/1e9, H_RRC[2], 'r', linewidth=1)
plt.plot(f/1e9, H_RRC_RRC[2], '--k', linewidth=2)
plt.plot(f/1e9, H_RC[2], 'b', linewidth=1)
plt.axvline(BR/2/1e9, linestyle='--', color='k')
plt.title(r'RC and RRC Frequency Response. $\beta = 0.5$')
plt.xlabel('Frequency [GHz]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend([r'$H_{RRC}$',r'$H_{RRC}\cdot H_{RRC}$',r'$H_{RC}$'])

plt.figure(figsize=(5,5))
plt.plot(f/1e9, 20*np.log10(H_RRC[2]), 'r', linewidth=1)
plt.plot(f/1e9, 20*np.log10(H_RC[2]), 'b', linewidth=1)
plt.plot(f/1e9, 20*np.log10(H_RRC_RRC[2]), '--k', linewidth=2)
plt.axvline(BR/2/1e9, linestyle='--', color='k')
plt.title(r'RC and RRC Frequency Response in dB. $\beta = 0.5$')
plt.xlabel('Frequency [GHz]')
plt.ylabel('Amplitude [dB]')
plt.grid(True)
plt.legend([r'$H_{RRC}$',r'$H_{RC}$',r'$H_{RRC}\cdot H_{RRC}$'])

plt.show()