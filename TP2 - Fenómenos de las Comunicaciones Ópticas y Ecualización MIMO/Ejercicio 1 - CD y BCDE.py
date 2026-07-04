import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.special import erfc

# -------------------------------------------------
# Basic TX QAM
# -------------------------------------------------

enable_plots = True

BR_v = np.array([30, 60, 90, 120]) * 1e9 # Baud rate [Hz]
# BR_v = np.array(120e9)
N_BR = len(BR_v)
N = 4
fs_v = N * BR_v

# CD
L_fiber_v = np.array([0, 60, 120, 240]) * 1e3         # longitud fibra [m]
# L_fiber_v = 120e3         # longitud fibra [m]
N_lfiber = len(L_fiber_v)
D = 20e-6               # ps/(nm km) -> convertido a s/(m^2)
lambda0 = 1550e-9       # longitud de onda [m]
c = 3e8                 # velocidad de la luz

# beta2 [s^2/m]
beta2 = -(D * lambda0**2) / (2*np.pi*c)

# NTAPS
def cd_len(L_fiber, BR, fs):
    CD_MAX = round(L_fiber * D, 3) 
    NTAPS_CD = round( (1.1 * fs * fs * CD_MAX * lambda0**2) / c)

    fs_bcde = fs # ESTO NO ES ASI EN LA IMPLEMENTACIÓN
    NTAPS_BCDE = round( (1.1 * fs * BR * CD_MAX * lambda0**2) / c)

    return CD_MAX, NTAPS_CD, NTAPS_BCDE

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

    # print(len(h))

    return h

def freq_response(h, fs):

    NFFT = 4*len(h)
    H = np.fft.fftshift(np.fft.fft(h, NFFT))
    f = np.fft.fftshift(np.fft.fftfreq(len(H), d=1/fs))

    # print(len(H))

    return f, H

# N taps
cd_max_m     = np.zeros((N_lfiber, N_BR))
ntaps_cd_m   = np.zeros((N_lfiber, N_BR))
ntaps_bcde_m = np.zeros((N_lfiber, N_BR))

c_td_m = np.zeros((N_lfiber, N_BR))
c_fd_m = np.zeros((N_lfiber, N_BR))

# CD
Nfft = 16*2048

h_cd_m    = np.zeros((N_lfiber, N_BR, Nfft), complex)
h_cd_zf_m = np.zeros((N_lfiber, N_BR, Nfft), complex)

H_cd_m    = np.zeros((N_lfiber, N_BR, 4*Nfft), complex)
H_eq_m    = np.zeros((N_lfiber, N_BR, 4*Nfft), complex)
f_cd_v    = np.zeros((N_BR, 4*Nfft))
f_eq_v    = np.zeros((N_BR, 4*Nfft))
phi_cd_m  = np.zeros((N_lfiber, N_BR, 4*Nfft))
phi_eq_m  = np.zeros((N_lfiber, N_BR, 4*Nfft))

for i in range(N_lfiber):
    for j in range(N_BR):
        cd_max_m[i][j], ntaps_cd_m[i][j], ntaps_bcde_m[i][j] = cd_len(L_fiber_v[i], BR_v[j], fs_v[j])

        # c_td_m[i][j] = 4 * ntaps_bcde_m[i][j] * ntaps_bcde_m[i][j]
        # c_fd_m[i][j] = 4 * ntaps_bcde_m[i][j] * (3*np.log2(ntaps_bcde_m[i][j]) + 5) + 1000

        h_cd_m[i][j]    = cd_fir(beta2,  L_fiber_v[i], fs_v[j])
        h_cd_zf_m[i][j] = cd_fir(beta2, -L_fiber_v[i], fs_v[j])

        f_cd_v[j], H_cd_m[i][j] = freq_response(h_cd_m[i][j]   , fs_v[j])
        f_eq_v[j], H_eq_m[i][j] = freq_response(h_cd_zf_m[i][j], fs_v[j])

        phi_cd_m[i][j] = -beta2*L_fiber_v[i]*(2*np.pi*f_cd_v[j])**2/2
        phi_eq_m[i][j] = +beta2*L_fiber_v[i]*(2*np.pi*f_cd_v[j])**2/2

# -------------------------------------------------
# Plots
# -------------------------------------------------

if enable_plots:

    # -------------------------------------------------
    # Fase vs long. de fibra
    # -------------------------------------------------

    plt.figure(figsize=(10,4))

    colors = ['red', 'green', 'orange', 'blue', 'purple', 'brown', 'pink']

    handles_cd = []
    handles_bcde = []
    labels = []

    # fase CD
    for i in range(len(L_fiber_v)):
        line, = plt.plot(f_cd_v[3]/1e9, phi_cd_m[i][3], color=colors[i])
        handles_cd.append(line)
        labels.append(f'L = {int(L_fiber_v[i]/1e3)} km')

    # fase BCDE
    for i in range(len(L_fiber_v)):
        line, = plt.plot(f_cd_v[3]/1e9, phi_eq_m[i][3], '--', color=colors[i])
        handles_bcde.append(line)

    # Leyenda CD
    legend1 = plt.legend(handles_cd, labels, loc='upper center', 
                        title='CD teorico', fontsize=9)
    plt.gca().add_artist(legend1)

    # Leyenda BCDE
    legend2 = plt.legend(handles_bcde, labels, loc='lower center', 
                        title='BCDE teorico', fontsize=9)
    plt.gca().add_artist(legend2)

    plt.ylabel("Fase [rad]")
    plt.xlabel("Frecuencia [GHz]")
    plt.title(f"Respuesta de Fase. BR = {int(BR_v[3]/1e9)} GBd")
    plt.grid()
    plt.tight_layout()

    # -------------------------------------------------
    # Fase vs BR
    # -------------------------------------------------

    plt.figure(figsize=(10,7))
    
    handles_cd = []
    handles_bcde = []
    labels = []

    for i in range(N_BR):
        plt.subplot(N_BR, 1, i + 1)

        if i == 0:
            # fase CD (líneas sólidas)
            line, = plt.plot(f_cd_v[i]/1e9, phi_cd_m[3][i], 'r', label='CD Teorico')
            # fase BCDE (líneas punteadas)
            line, = plt.plot(f_cd_v[i]/1e9, phi_eq_m[3][i], '--r', label='BCDE Teorico')
            plt.legend()

        else:
            # fase CD (líneas sólidas)
            line, = plt.plot(f_cd_v[i]/1e9, phi_cd_m[3][i], 'r')
            # fase BCDE (líneas punteadas)
            line, = plt.plot(f_cd_v[i]/1e9, phi_eq_m[3][i], '--r')

        plt.title(rf'$BR = {int(BR_v[i]/1e9)}$ GBd')
        plt.ylabel("Fase [rad]")
        plt.xlim([-250, 250])
        plt.ylim([-8000, 8000])
        if i < N_BR - 1: plt.gca().set_xticklabels([])
        plt.grid()

    plt.xlabel("Frecuencia [GHz]")
    plt.tight_layout()

    # -------------------------------------------------
    # Ntaps del BCDE vs long. de fibra vs BR
    # -------------------------------------------------

    # plt.figure(figsize=(6.6,4))
    plt.figure(figsize=(4,4))

    for i in range(N_BR):
        plt.plot(L_fiber_v/1e3, ntaps_bcde_m[:, i], label=f'BR = {int(BR_v[i]/1e9)} GBd')

    plt.ylabel("N° Taps")
    plt.xlabel("Long. de Fibra [km]")
    plt.legend()
    plt.title("Nro. de Taps del BCDE")
    plt.xlim((-5,80))
    plt.ylim((-2,45))
    plt.grid()
    plt.tight_layout()
    
    # L fibra minimo para pasar al dominio de frecuencia:
    # 120 gbd: 2.8km
    # 90 gbd:  5km
    # 60 gbd:  11.4km
    # 30 gbd:  45.7km

    # -------------------------------------------------
    # Complejidad de implementacion vs Nro. de taps
    # -------------------------------------------------

    ntaps_v = np.arange(1, 80)
    c_td_v = np.zeros(len(ntaps_v))
    c_fd_v = np.zeros(len(ntaps_v))
    
    for i in range(len(ntaps_v)):
        c_td_v[i] = 4 * ntaps_v[i] * ntaps_v[i]
        c_fd_v[i] = 4 * ntaps_v[i] * (3*np.log2(ntaps_v[i]) + 5) + 1000

    plt.figure(figsize=(4,4))

    plt.plot(ntaps_v, c_td_v/1e3, label='Dominio del Tiempo')
    plt.plot(ntaps_v, c_fd_v/1e3, label='Dominio de la Frecuencia')

    plt.ylabel(r"N° Multiplicaciones ($\times 10^3$)")
    plt.xlabel("N° Taps")
    plt.legend()
    plt.title("Complejidad del BCDE")
    plt.grid()
    plt.tight_layout()
    plt.show()