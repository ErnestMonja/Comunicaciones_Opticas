import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.special import erfc

# -------------------------------------------------
# Basic TX QAM
# -------------------------------------------------

enable_plots = True

BR = 120e9
N = 4
fs = N * BR

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
h_cd_zf = cd_fir(beta2, -L_fiber, fs)

# -------------------------------------------------
# Plots
# -------------------------------------------------

def freq_response(h, fs):

    NFFT = 4*len(h)
    H = np.fft.fftshift(np.fft.fft(h, NFFT))
    f = np.fft.fftshift(np.fft.fftfreq(len(H), d=1/fs))

    return f, H

if enable_plots:
    
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
