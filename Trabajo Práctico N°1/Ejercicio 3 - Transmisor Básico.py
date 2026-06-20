import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, welch

plt.close("all")

# -------------------------------------------------
# Parámetros
# -------------------------------------------------
M = 4               # Modulación PAM-4 (Ejercicio 3)
L = int(1e4)        # Longitud de simulación
BR = 32e9           # Symbol rate
N = 4               # Factor de sobremuestreo
rolloff = 0.5       # Coeficiente de Beta
h_taps = 101        # Cantidad de taps.

T = 1/BR
fs = N * BR
Ts = 1/fs



# -------------------------------------------------
# Funciones de filtro
# -------------------------------------------------
def round_odd(n):
    n = int(np.round(n))
    if n % 2 == 0:
        n += 1
    return n



# -------------------------------------------------
# Funciones de RC y RRC
# -------------------------------------------------
def raised_cosine(fc, fs, rolloff, n_taps, t0=0):
    """Filtro Coseno Realzado (RC)"""
    Ts = 1 / fs
    rolloff = rolloff + 0.0001
    T = 1 / fc
    n_taps = round_odd(n_taps)
    n = np.arange(-(n_taps - 1)//2, (n_taps - 1)//2 + 1)
    t_v = n * Ts + t0
    tn_v = t_v * 2 / T
    
    h_v = np.sinc(tn_v) * np.cos(np.pi * rolloff * tn_v) / (1 - (2 * rolloff * tn_v)**2)
    h_v = h_v / np.sum(h_v)
    return h_v

def root_raised_cosine(fc, fs, rolloff, n_taps, t0=0):
    """Filtro Raíz de Coseno Realzado (RRC) - Tomado del script del profe"""
    Ts = 1 / fs
    rolloff = rolloff + 0.0001
    T = 1 / fc
    n_taps = round_odd(n_taps)
    n = np.arange(-(n_taps - 1)//2, (n_taps - 1)//2 + 1)
    t_v = n * Ts + t0
    tn_v = t_v * 2 / T

    numerator = np.sin(np.pi * (1 - rolloff) * tn_v) + 4 * rolloff * tn_v * np.cos(np.pi * (1 + rolloff) * tn_v)
    denominator = np.pi * tn_v * (1 - (4 * rolloff * tn_v)**2)
    h_v = numerator / denominator

    center = (n_taps - 1) // 2
    h_v[center] = (1 + rolloff * (4/np.pi - 1))
    h_v = h_v / np.sum(h_v) # Normalización para TX
    return h_v



# -------------------------------------------------
# Generador de bits y Mapper para PAM-M
# -------------------------------------------------
# Generamos símbolos PAM directamente (ej: -3, -1, 1, 3)
ak = np.random.choice([-3, -1, 1, 3], L)



# -------------------------------------------------
# Upsampling
# -------------------------------------------------
xup = np.zeros(L*N)
xup[::N] = ak



# -------------------------------------------------
# Filtrado (Usando lfilter y compensando delay)
# -------------------------------------------------
h_rc = raised_cosine(BR/2, fs, rolloff, h_taps)
h_rrc = root_raised_cosine(BR/2, fs, rolloff, h_taps)

h_delay = (h_taps - 1) // 2  

# Filtrado y alineación temporal concatenando ceros
s_t_rc = lfilter(h_rc, 1, np.concatenate([xup, np.zeros(h_delay)]))[h_delay:]
s_t_rrc = lfilter(h_rrc, 1, np.concatenate([xup, np.zeros(h_delay)]))[h_delay:]



# -------------------------------------------------
# PUNTOS 1 y 2: Gráficos en el Dominio del Tiempo
# -------------------------------------------------
mostrar = 20
t_simbolos = np.arange(mostrar)
t_senal = np.arange(mostrar * N) / N

plt.figure(figsize=(14, 5))

# Señal con Filtro RC
plt.subplot(1, 2, 1)
plt.plot(t_senal, s_t_rc[:mostrar * N], label='s(t) [RC]')
plt.stem(t_simbolos, ak[:mostrar], linefmt='r--', markerfmt='ro', basefmt='k', label='Símbolos (ak)')
plt.title("Alineación con Filtro RC (Cero ISI)")
plt.xlabel("Tiempo Normalizado (t/T)")
plt.grid(True); plt.legend()

# Señal con Filtro RRC
plt.subplot(1, 2, 2)
plt.plot(t_senal, s_t_rrc[:mostrar * N], label='s(t) [RRC]')
plt.stem(t_simbolos, ak[:mostrar], linefmt='r--', markerfmt='ro', basefmt='k', label='Símbolos (ak)')
plt.title("Alineación con Filtro RRC (Con ISI)")
plt.xlabel("Tiempo Normalizado (t/T)")
plt.grid(True); plt.legend()

plt.tight_layout()
plt.show()



# -------------------------------------------------
# PUNTOS 3 y 4: PSD usando método de Welch
# -------------------------------------------------
# Calculo de PSD a la entrada
f_in, Pxx_in = welch(xup, fs, nperseg=1024, return_onesided=False)
f_in = np.fft.fftshift(f_in)
Pxx_in_dB = 10 * np.log10(np.fft.fftshift(Pxx_in))
Pxx_in_dB -= np.max(Pxx_in_dB) # Normalizamos a 0 dB

# Calculo de PSD a la salida del filtro RC
f_out, Pxx_out = welch(s_t_rc, fs, nperseg=1024, return_onesided=False)
f_out = np.fft.fftshift(f_out)
Pxx_out_dB = 10 * np.log10(np.fft.fftshift(Pxx_out))
Pxx_out_dB -= np.max(Pxx_out_dB) # Normalizamos a 0 dB

# Respuesta en frecuencia teórica |H(f)|^2
H_f = np.fft.fft(h_rc, 1024)
H_f_mag_sq_dB = 20 * np.log10(np.abs(H_f))
H_f_mag_sq_dB = np.fft.fftshift(H_f_mag_sq_dB)
H_f_mag_sq_dB -= np.max(H_f_mag_sq_dB)

plt.figure(figsize=(14, 5))

# PSD Entrada
plt.subplot(1, 2, 1)
plt.plot(f_in / 1e9, Pxx_in_dB)
plt.title("Punto 3: PSD a la Entrada del Filtro")
plt.xlabel("Frecuencia [GHz]")
plt.ylabel("Densidad Espectral [dB]")
plt.ylim([-50, 5])
plt.grid(True)

# PSD Salida vs Teórica
plt.subplot(1, 2, 2)
plt.plot(f_out / 1e9, Pxx_out_dB, label='PSD Salida Welch', lw=2)
plt.plot(f_in / 1e9, H_f_mag_sq_dB, 'r--', label='|H(f)|²', lw=2)
plt.title("Punto 4: PSD a la Salida vs |H(f)|²")
plt.xlabel("Frecuencia [GHz]")
plt.legend()
plt.ylim([-50, 5])
plt.grid(True)

plt.tight_layout()
plt.show()