import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift

plt.close("all")

# -------------------------------------------------
# Parámetros iniciales.
# -------------------------------------------------
BR = 32e9                   # Symbol rate: 32 GBd
N = 4                       # Factor de sobremuestreo.
fs = N * BR                 # Frecuencia de muestreo (4 * BR).
h_taps = 101                # Cantidad de taps.
betas = [0.0, 0.5, 1.0]     # Valores de rolloff (beta) variables solicitados.



# -------------------------------------------------
# Función round_odd()
# -------------------------------------------------
def round_odd(n):
    n = int(np.round(n))
    if n % 2 == 0:
        n += 1
    return n



# -------------------------------------------------
# 1. Función Raised Cosine (Adaptada).
# -------------------------------------------------
def raised_cosine(symbol_rate, fs, rolloff, n_taps):
    # Se suma un valor minúsculo para evitar divisiones por cero en el denominador.
    rolloff = rolloff + 1e-9  
    
    Ts = 1 / fs               # Período de muestreo del filtro,
    T  = 1 / symbol_rate      # Tiempo de símbolo,
    
    n_taps = round_odd(n_taps)
    
    # Vector de índices y vector de tiempos,
    n = np.arange(-(n_taps - 1)//2, (n_taps - 1)//2 + 1)
    t_v = n * Ts
    
    # Variable temporal normalizada respecto al tiempo de símbolo (t/T)
    tn_v = t_v / T 
    
    # Ecuación del Raised Cosine (Nota: np.sinc en numpy ya está multiplicada por pi: sinc(x) = sin(pi*x)/(pi*x))
    numerador   = np.sinc(tn_v) * np.cos(np.pi * rolloff * tn_v)
    denominador = 1 - (2 * rolloff * tn_v)**2
    
    h_v = numerador / denominador
    
    # Normalización para ganancia unitaria
    h_v = h_v / np.sum(h_v)
    
    return t_v, h_v



# -------------------------------------------------
# 2. Generación y Plot en el Dominio del Tiempo.
# -------------------------------------------------
plt.figure(figsize=(8, 5))

# quedan guardadas las respuestas al impulso para reusarlas en el punto 3.
respuestas_impulso = []
T_simbolo = 1 / BR

for beta in betas:
    t_v, h_rc = raised_cosine(BR, fs, beta, h_taps)
    respuestas_impulso.append(h_rc)
    
    # Ploteamos normalizando el eje X al tiempo de símbolo (t/T) para apreciar los cruces.
    plt.plot(t_v / T_simbolo, h_rc, linewidth=1.5, label=f'$\\beta$ = {beta}')

plt.title('Punto 2: Respuesta al Impulso del Filtro Coseno Elevado')
plt.xlabel('Tiempo normalizado ($t/T$)')
plt.ylabel('Amplitud')
plt.xlim([-4, 4]) # Zoom en los lóbulos principales
plt.grid(True)
plt.legend()



# -------------------------------------------------
# 3. FFTs y Plot en el Dominio de la Frecuencia.
# -------------------------------------------------
NFFT = 2048
# Vector de frecuencias
f = np.arange(-NFFT/2, NFFT/2) * fs / NFFT

plt.figure(figsize=(8, 5))

for i, beta in enumerate(betas):
    h_rc = respuestas_impulso[i]
    
    # Calculamos la magnitud de la FFT y centramos el espectro.
    H_RC = fftshift(np.abs(fft(h_rc, NFFT)))
    
    # Normalizamos el máximo a 0 [dB] para realizar un mejor comparación.
    H_RC_dB = 20 * np.log10(H_RC / np.max(H_RC))
    
    plt.plot(f / 1e9, H_RC_dB, linewidth=1.5, label=f'$\\beta$ = {beta}')

# Línea vertical marcando la frecuencia de Nyquist (BR / 2)
plt.axvline(BR / 2 / 1e9, linestyle='--', color='k', label='Nyquist ($R_s/2$)')

plt.title('Punto 3: Respuesta en Frecuencia (Magnitud)')
plt.xlabel('Frecuencia [GHz]')
plt.ylabel('Magnitud [dB]')
plt.xlim([0, 35])       # Se muestran solo las frecuencias positivas.
plt.ylim([-60, 5])
plt.grid(True)
plt.legend()

plt.show()