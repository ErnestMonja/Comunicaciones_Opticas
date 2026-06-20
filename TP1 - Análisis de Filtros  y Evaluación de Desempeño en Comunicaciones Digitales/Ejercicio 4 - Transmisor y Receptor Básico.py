import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, welch
from scipy.special import erfc

plt.close("all")

# --------------------------------------------------------------------------------------------------
# FUNCIONES BASE 
# --------------------------------------------------------------------------------------------------
# Modulador M-QAM
def qammod(symbols, M):
    # m es la cantidad de niveles por eje (I y Q).
    # Para QAM cuadrada se cumple que M = m^2 (por ejemplo: 16-QAM → m=4).
    m = int(np.sqrt(M))

    # Calcula la componente real (I) del símbolo:
    # symbols % m da la posición horizontal dentro de la grilla.
    # La expresión genera niveles igualmente espaciados: {-m+1, ..., m-1}.
    # sym=0,1,2,3   sym%m=0,1,0,1   re=-1,1,-1,1
    # sym=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 niv=0,1,2,3 x4 niv^niv>1=0,1,3,2 x4 re=-3,-1,3,1 x4
    niv = (symbols % m)
    re = 2 * (niv^(niv>>1)) - m + 1
    
    # Calcula la componente imaginaria (Q):
    # symbols // m determina la fila vertical dentro de la grilla.
    # sym=0,1,2,3   sym//m=0,0,1,1  im=-1,-1,1,1    
    # sym=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 niv=0x4,1x4,2x4,3x4 niv^niv>1=0x4,1x4,3x4,2x4
    # im=-3x4,-1x4,3x4,1x4
    niv = (symbols // m)
    im = 2 * (niv^(niv>>1)) - m + 1
    
    # Forma el símbolo complejo I + jQ .
    return (re + 1j*im)


# Cálculo teórico de BER para M-QAM
def ber_theoretical(EbNo_db, M):
    k = np.log2(M)
    EbNo = 10**(EbNo_db/10)
    return (4/k)*(1-1/np.sqrt(M))*0.5*erfc(np.sqrt(3*k*EbNo/(2*(M-1))))


# Filtro RRC con normalización de energía (Trabajamos con índices discretos (n) para evitar los bugs de tolerancia de punto flotante de Python).
def root_raised_cosine(BR, fs, rolloff, n_taps):
    sps = int(fs / BR)                          # Muestras por símbolo (Equivale a N)
    n = np.arange(n_taps) - (n_taps - 1) // 2
    h = np.zeros(n_taps)
    
    for i in range(n_taps):
        if n[i] == 0:
            h[i] = 1.0 - rolloff + (4 * rolloff / np.pi)
        elif rolloff != 0 and np.isclose(np.abs(n[i]), sps / (4 * rolloff), atol=1e-5):
            h[i] = (rolloff / np.sqrt(2)) * (((1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff))) + ((1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff))))
        else:
            num = np.sin(np.pi * n[i] * (1 - rolloff) / sps) + 4 * rolloff * (n[i] / sps) * np.cos(np.pi * n[i] * (1 + rolloff) / sps)
            den = np.pi * (n[i] / sps) * (1 - (4 * rolloff * n[i] / sps)**2)
            h[i] = num / den
            
    return h / np.sqrt(np.sum(h**2))            # Normalizado en energía.


# Genera el Diagrama de Ojo alineado.
def eye_diagram(s_t, sps, h_taps, num_symbols=150):
    plt.figure(figsize=(7, 5))
    delay = (h_taps - 1) // 2
    start_sym = 100 
    t_axis = np.arange(-sps, sps + 1) / sps     # Eje X normalizado (-1 a 1).
    
    for k in range(start_sym, start_sym + num_symbols):
        peak_idx = delay + k * sps
        segment = s_t[peak_idx - sps : peak_idx + sps + 1]
        if len(segment) == 2 * sps + 1:
            plt.plot(t_axis, segment.real, 'b-', alpha=0.3)
            
    plt.title('1. Diagrama de Ojo a la salida del TX (Parte Real)')
    plt.xlabel('Tiempo normalizado (t/T)')
    plt.ylabel('Amplitud')
    plt.grid(True)


# Simula la cadena TX -> Canal AWGN -> RX entera
def simular_sistema(M, L, BR, N, rolloff, h_taps, EbNo_db, graficar=False):
    fs = N * BR
    k = np.log2(M)
    
    # 1. Transmisor
    x_tx = np.random.randint(0, M, L)
    ak = qammod(x_tx, M)
    Es = 2 * (M - 1) / 3
    
    xup = np.zeros(L*N, dtype=complex)
    xup[::N] = ak
    
    h_rrc = root_raised_cosine(BR, fs, rolloff, h_taps)
    s_t = lfilter(h_rrc, 1, xup)
    
    # 2. Canal AWGN
    EbNo_lin = 10**(EbNo_db / 10)
    N0 = Es / (k * EbNo_lin)
    sigma2 = N0 
    
    noise = np.sqrt(sigma2/2) * (np.random.randn(len(s_t)) + 1j * np.random.randn(len(s_t)))
    r_t = s_t + noise
    
    # 3. Receptor (Filtro Apareado)
    y_t = lfilter(h_rrc, 1, r_t)
    
    # Alineación (el pico se forma en: h_taps - 1)
    delay = h_taps - 1
    rx_aligned = y_t[delay :: N] 
    rx_aligned = rx_aligned[:L-delay//N] 
    x_tx_aligned = x_tx[:len(rx_aligned)]
    
    # 4. Slicer
    const = qammod(np.arange(M), M)
    idx_rx = np.argmin(np.abs(rx_aligned[:, None] - const), axis=1)
    
    # Cálculo de BER (Aproximación por SER / k para calcular la teórica)
    errores = np.sum(x_tx_aligned != idx_rx)
    ber_simulada = errores / (len(x_tx_aligned) * k)
    
    # ------------- GRÁFICOS -------------
    if graficar:
        eye_diagram(s_t, N, h_taps)
        
        # Welch PSDs (Normalizados al pico)
        f_s, Pxx_s = welch(s_t, fs, nperseg=2048, return_onesided=False)
        f_r, Pxx_r = welch(r_t, fs, nperseg=2048, return_onesided=False)
        f_y, Pxx_y = welch(y_t, fs, nperseg=2048, return_onesided=False)
        
        f_shift = np.fft.fftshift(f_s) / 1e9
        P_s_db = 10*np.log10(np.fft.fftshift(Pxx_s) / np.max(Pxx_s))
        P_r_db = 10*np.log10(np.fft.fftshift(Pxx_r) / np.max(Pxx_r))
        P_y_db = 10*np.log10(np.fft.fftshift(Pxx_y) / np.max(Pxx_y))
        
        plt.figure(figsize=(7, 5))
        plt.plot(f_shift, P_s_db, label='PSD TX', lw=2)
        plt.plot(f_shift, P_r_db, label='PSD Entrada RX (con AWGN)', alpha=0.7)
        plt.title("2. PSD: Salida TX vs Entrada RX")
        plt.xlabel("Frecuencia [GHz]"); plt.ylabel("Magnitud Normalizada [dB]")
        plt.ylim([-50, 5])
        plt.grid(True); plt.legend()
        
        plt.figure(figsize=(7, 5))
        plt.plot(f_shift, P_r_db, label='PSD Entrada RX', lw=2)
        plt.plot(f_shift, P_y_db, label='PSD Salida Filtro RX', alpha=0.8)
        plt.title("3. PSD: Entrada RX vs Salida Filtro RX (Limpieza)")
        plt.xlabel("Frecuencia [GHz]"); plt.ylabel("Magnitud Normalizada [dB]")
        plt.ylim([-50, 5])
        plt.grid(True); plt.legend()
        
        plt.figure(figsize=(6, 6))
        plt.scatter(rx_aligned.real, rx_aligned.imag, s=2, alpha=0.5, c='b', label='Recibidos')
        plt.scatter(const.real, const.imag, c='r', marker='x', s=100, label='Ideales')
        plt.title(f"4. Constelación (Eb/N0 = {EbNo_db} dB)")
        lim = np.max(np.abs(const)) + 2
        plt.xlim([-lim, lim]); plt.ylim([-lim, lim])
        plt.grid(True); plt.legend()
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(rx_aligned.real, bins=50, color='b', alpha=0.7)
        plt.title("5a. Histograma - Parte Real")
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.hist(rx_aligned.imag, bins=50, color='g', alpha=0.7)
        plt.title("5b. Histograma - Parte Imaginaria")
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    return ber_simulada



# --------------------------------------------------------------------------------------------------
# PARTE A: Gráficos (Eb/N0 de 15 [dB] para ver el ojo limpio)
# --------------------------------------------------------------------------------------------------
print("--- Corriendo PARTE A: Generando Gráficos ---")
simular_sistema(M=16, L=20000, BR=32e9, N=4, rolloff=0.5, h_taps=101, EbNo_db=15, graficar=True)



# --------------------------------------------------------------------------------------------------
# PARTE B: Curvas de BER (Saltos a 1.5 [dB] para que corra más rápido)
# --------------------------------------------------------------------------------------------------
print("\n--- Corriendo PARTE B: Curvas de BER ---")
ebno_range = np.arange(0, 15.5, 1.5)
L_ber = 50000 

ber_sim_qpsk = []; ber_teo_qpsk = []
ber_sim_qam16 = []; ber_teo_qam16 = []

for ebno in ebno_range:
    print(f"Simulando Eb/No = {ebno} dB ...")
    ber_sim_qpsk.append(simular_sistema(4, L_ber, 32e9, 4, 0.5, 101, ebno))
    ber_teo_qpsk.append(ber_theoretical(ebno, 4))
    
    ber_sim_qam16.append(simular_sistema(16, L_ber, 32e9, 4, 0.5, 101, ebno))
    ber_teo_qam16.append(ber_theoretical(ebno, 16))

plt.figure(figsize=(8, 6))
plt.semilogy(ebno_range, ber_teo_qpsk, 'r-', lw=2, label='QPSK Teórica ')
plt.semilogy(ebno_range, ber_sim_qpsk, 'ro', label='QPSK Simulada')

plt.semilogy(ebno_range, ber_teo_qam16, 'b-', lw=2, label='QAM-16 Teórica')
plt.semilogy(ebno_range, ber_sim_qam16, 'bs', label='QAM-16 Simulada')

plt.ylim([1e-6, 1])
plt.title("Curvas de BER vs $E_b/N_0$")
plt.xlabel("$E_b/N_0$ [dB]")
plt.ylabel("Bit Error Rate (BER)")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()

print("¡Simulaciones finalizadas!")
