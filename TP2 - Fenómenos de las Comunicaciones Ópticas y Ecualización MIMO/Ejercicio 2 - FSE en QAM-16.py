import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, firwin
from scipy.special import erfc
from scipy.interpolate import interp1d
from numba import njit
import warnings

warnings.filterwarnings('ignore')
plt.close("all")



# --------------------------------------------------------------------------------------------------
# Funciones Base y comunicaciones
# --------------------------------------------------------------------------------------------------
def qammod(symbols, M):
    m = int(np.sqrt(M))
    niv = (symbols % m)
    re = 2 * (niv ^ (niv >> 1)) - m + 1
    niv = (symbols // m)
    im = 2 * (niv ^ (niv >> 1)) - m + 1
    return (re + 1j * im)

def qamdemod(rx_symbols, M):
    constellation = qammod(np.arange(M), M)
    distances = np.abs(rx_symbols[:, np.newaxis] - constellation)
    return np.argmin(distances, axis=1)

def ber_theoretical(EbNo_db, M):
    k = np.log2(M)
    EbNo = 10**(EbNo_db / 10)
    return (4/k) * (1 - 1/np.sqrt(M)) * 0.5 * erfc(np.sqrt(3 * k * EbNo / (2 * (M - 1))))

def root_raised_cosine(BR, fs, rolloff, n_taps):
    sps = int(fs / BR)
    n = np.arange(n_taps) - (n_taps - 1) // 2
    h = np.zeros(n_taps)
    for i in range(n_taps):
        if n[i] == 0:
            h[i] = 1.0 - rolloff + (4 * rolloff / np.pi)
        elif rolloff != 0 and np.isclose(np.abs(n[i]), sps / (4 * rolloff), atol=1e-5):
            h[i] = (rolloff / np.sqrt(2)) * (((1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff))) +
                                             ((1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff))))
        else:
            num = np.sin(np.pi * n[i] * (1 - rolloff) / sps) + 4 * rolloff * (n[i] / sps) * np.cos(np.pi * n[i] * (1 + rolloff) / sps)
            den = np.pi * (n[i] / sps) * (1 - (4 * rolloff * n[i] / sps)**2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h**2))



# --------------------------------------------------------------------------------------------------
# Ecualizador FSE: Acelerado con librería "Numba"
# --------------------------------------------------------------------------------------------------
@njit
def fse_core(y, w, Ntaps, mu, N_sps, R_CMA, cma_limit, constellation):
    L = len(y)
    L_out = L // N_sps
    out_sym = np.zeros(L_out, dtype=np.complex128)
    err_hist = np.zeros(L_out, dtype=np.float64) # Agregamos historial de error
    
    for i in range(L_out):
        idx_in = i * N_sps
        if idx_in + Ntaps > L:
            break
            
        buffer = y[idx_in : idx_in + Ntaps][::-1] 
        out = np.dot(w, buffer)
        out_sym[i] = out
        
        if i % 200 == 0:
            if np.isnan(out.real) or np.abs(out) > 100:
                out_sym[0] = np.nan + 1j * np.nan
                return out_sym, err_hist
                
        # Etapa 1: CMA
        if i < cma_limit:
            e = out * (np.abs(out)**2 - R_CMA)
            if np.abs(e) > 50: 
                e = 50 * e / np.abs(e) 
            err_hist[i] = np.abs(np.abs(out)**2 - R_CMA)**2
        # Etapa 2: Decision Directed (DD)
        else:
            min_dist = 1e9
            dec = out
            for c in constellation:
                dist = np.abs(out - c)
                if dist < min_dist:
                    min_dist = dist
                    dec = c
            e = out - dec
            err_hist[i] = np.abs(e)**2
            
        pwr = np.sum(np.abs(buffer)**2) + 1e-8
        w = w - (mu / pwr) * e * np.conj(buffer)
        
    return out_sym, err_hist

def fse_equalizer(y, M, Ntaps, mu, N_sps, h_rrc, cma_frac=0.25):
    w = np.zeros(Ntaps, dtype=complex)
    h_taps = len(h_rrc)
    start_idx = (Ntaps - h_taps) // 2
    if start_idx >= 0:
        w[start_idx : start_idx + h_taps] = h_rrc
    else:
        w[:] = h_rrc[-start_idx : -start_idx + Ntaps]
        
    constellation = qammod(np.arange(M), M).astype(np.complex128)
    R_CMA = np.mean(np.abs(constellation)**4) / np.mean(np.abs(constellation)**2)
    cma_limit = int((len(y) // N_sps) * cma_frac)
    
    out_sym, err_hist = fse_core(y.astype(np.complex128), w, Ntaps, mu, N_sps, R_CMA, cma_limit, constellation)
    
    if np.isnan(out_sym[0].real):
        return None, None
    return out_sym, err_hist



# --------------------------------------------------------------------------------------------------
# Simulador Principal
# --------------------------------------------------------------------------------------------------
def simular_fse(M, L, BR, N, rolloff, h_taps, EbNo_db, Ntaps, mu, BW_limit=False, return_full=False):
    fs = N * BR
    k = np.log2(M)
    
    x_tx = np.random.randint(0, M, L)
    ak = qammod(x_tx, M)
    Es = 2 * (M - 1) / 3
    xup = np.zeros(L * N, dtype=complex)
    xup[::N] = ak
    
    h_rrc = root_raised_cosine(BR, fs, rolloff, h_taps)
    s_t = lfilter(h_rrc, 1, xup)
    
    if BW_limit:
        h_bw = firwin(61, 0.70) 
        s_t = lfilter(h_bw, 1, s_t)
        
    EbNo_lin = 10**(EbNo_db / 10)
    N0 = Es / (k * EbNo_lin)
    noise = np.sqrt(N0 * N / 2) * (np.random.randn(len(s_t)) + 1j * np.random.randn(len(s_t)))
    r_t = s_t + noise
    
    rx_eq, err_hist = fse_equalizer(r_t, M, Ntaps, mu, N_sps=N, h_rrc=h_rrc, cma_frac=0.3)
    
    if rx_eq is None:
        if return_full: return 0.5, None, None, None
        return 0.5 
    
    # Sincronización y cálculo de BER
    trim_idx = int(len(rx_eq) * 0.5) 
    rx_eq_trim = rx_eq[trim_idx:]
    x_tx_trim = x_tx[trim_idx:len(rx_eq_trim) + trim_idx]
    
    tx_sync = ak[trim_idx : trim_idx+800]
    rx_sync = rx_eq_trim[:800]
    corr = np.abs(np.correlate(rx_sync, tx_sync, mode='full'))
    delay = np.argmax(corr) - (len(tx_sync) - 1)
    
    if delay > 0:
        rx_aligned = rx_eq_trim[delay:]
        tx_aligned = x_tx_trim[:len(rx_aligned)]
    elif delay < 0:
        rx_aligned = rx_eq_trim[:delay]
        tx_aligned = x_tx_trim[-delay:len(rx_aligned)-delay]
    else:
        rx_aligned = rx_eq_trim
        tx_aligned = x_tx_trim
        
    tx_ideal = qammod(tx_aligned, M)
    c = np.sum(tx_ideal * np.conj(rx_aligned)) / (np.sum(np.abs(rx_aligned)**2) + 1e-12)
    rx_aligned = rx_aligned * c
            
    idx_rx = qamdemod(rx_aligned, M)
    errores = np.sum(tx_aligned != idx_rx)
    ber = errores / (max(len(tx_aligned), 1) * k)
    
    if return_full:
        # Extraer señal r_t downsampleada para mostrar la constelación mala
        r_t_down = r_t[::N]
        return ber, r_t_down, rx_aligned, err_hist
    return ber



# --------------------------------------------------------------------------------------------------
# Barrido y obtención de la penalidad:
# --------------------------------------------------------------------------------------------------
def obtener_penalidad(M, Ntaps, mu, BW_limit):
    ebno_range = np.arange(10, 22, 1.0) 
    ber_sim = []
    
    for ebno in ebno_range:
        ber = simular_fse(M=16, L=150000, BR=32e9, N=2, rolloff=0.5, h_taps=101, 
                          EbNo_db=ebno, Ntaps=Ntaps, mu=mu, BW_limit=BW_limit)
        ber_sim.append(ber)
    
    ber_sim = np.array(ber_sim)
    valid_idx = ber_sim > 0
    b_sim = ber_sim[valid_idx]
    e_rng = ebno_range[valid_idx]
    
    if len(b_sim) < 2 or np.min(b_sim) > 0.05:
        return 15.0
        
    b_log = np.log10(b_sim)
    idx_sort = np.argsort(b_log)
    b_log_s = b_log[idx_sort]
    e_rng_s = e_rng[idx_sort]
    
    _, unique_idx = np.unique(b_log_s, return_index=True)
    b_log_u = b_log_s[unique_idx]
    e_rng_u = e_rng_s[unique_idx]
    
    if len(b_log_u) < 2:
        return 15.0
        
    try:
        f_sim = interp1d(b_log_u, e_rng_u, kind='linear', fill_value="extrapolate")
        ebno_sim_1e3 = f_sim(-3.0) 
    except Exception:
        return 15.0
    
    ebno_teo_range = np.linspace(8, 22, 100)
    ber_teo = np.array([ber_theoretical(e, M) for e in ebno_teo_range])
    ber_teo_log = np.log10(ber_teo + 1e-12)
    f_teo = interp1d(ber_teo_log, ebno_teo_range, kind='linear', fill_value="extrapolate")
    ebno_teo_1e3 = f_teo(-3.0)
    
    penalty = ebno_sim_1e3 - ebno_teo_1e3
    return max(0.0, min(penalty, 15.0))

# --------------------------------------------------------------------------------------------------
# Ejecución Final: 
# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    M_qam = 16
    
    # ----------------------------------------------------------------------------------------------
    # PARTE 1: Barrido de la Penalidad (Parte Lenta)
    # ----------------------------------------------------------------------------------------------
    print("--- Iniciando Barrido Pesado ---")
    ntaps_list = [31, 63, 127, 255]
    mu_exponents = np.arange(-13, -3, 1.0)
    mu_list = 2.0**mu_exponents

    fig1, axes = plt.subplots(1, 2, figsize=(16, 6))

    for bw_idx, BW_lim in enumerate([False, True]):
        ax = axes[bw_idx]
        title = "Con Limitación de BW" if BW_lim else "Sin Limitación de BW"
        
        for ntaps in ntaps_list:
            penalties = []
            for mu_exp, mu in zip(mu_exponents, mu_list):
                pen = obtener_penalidad(M_qam, ntaps, mu, BW_lim)
                penalties.append(pen)
                
            ax.plot(mu_exponents, penalties, marker='o', label=f'Ntaps = {ntaps}')
            
        ax.set_title(title)
        ax.set_xlabel(r'$\log_2(\mu)$')
        ax.set_ylabel(r'Penalidad de SNR @ BER = $10^{-3}$ [dB]')
        ax.grid(True, linestyle='--')
        ax.legend()
        ax.set_ylim([-0.5, 10]) 

    fig1.tight_layout()
    
    # ----------------------------------------------------------------------------------------------
    # PARTE 2: Análisis Profundo (Constelaciones y Curva de Aprendizaje)
    # ----------------------------------------------------------------------------------------------
    print("\n--- Generando Gráficas de Análisis (Constelaciones y Convergencia) ---")
    
    # Elegimos un caso robusto: 63 taps, mu óptimo, con limitación de BW a 20 dB de Eb/No.
    ntaps_test = 63
    mu_test = 2**(-7)
    L_test = 150000
    cma_fraction = 0.3      # 30% del tiempo en CMA.
    
    _, rx_raw, rx_eq, err_hist = simular_fse(M=16, L=L_test, BR=32e9, N=2, 
                                             rolloff=0.5, h_taps=101, EbNo_db=20, 
                                             Ntaps=ntaps_test, mu=mu_test, 
                                             BW_limit=True, return_full=True)

    # Figura 2: Constelaciones.
    fig2, (ax_raw, ax_eq) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Tomamos los últimos 5000 símbolos para ver el estado estable.
    ax_raw.scatter(rx_raw[-5000:].real, rx_raw[-5000:].imag, alpha=0.3, color='red', s=5)
    ax_raw.set_title('Señal Recibida (Canal Limitado + Ruido)')
    ax_raw.grid(True, linestyle='--')
    ax_raw.axis('equal')
    
    ax_eq.scatter(rx_eq[-5000:].real, rx_eq[-5000:].imag, alpha=0.3, color='blue', s=5)
    ax_eq.set_title('Señal Post-Ecualización (FSE 63 Taps)')
    ax_eq.grid(True, linestyle='--')
    ax_eq.axis('equal')
    fig2.tight_layout()

    # Figura 3: Curva de Convergencia.
    fig3, ax_err = plt.subplots(figsize=(10, 5))
    
    # Suavizamos la curva de error con una media móvil para que se vea la tendencia.
    window_size = 500
    err_smooth = np.convolve(err_hist, np.ones(window_size)/window_size, mode='valid')
    
    ax_err.plot(10 * np.log10(err_smooth + 1e-12), color='purple')
    
    # Marcamos el punto exacto donde el código pasa de CMA a DD.
    cma_switch_idx = int(L_test * cma_fraction)
    ax_err.axvline(x=cma_switch_idx, color='k', linestyle='--', label=r'Conmutación CMA $\rightarrow$ DD')
    
    ax_err.set_title(f'Curva de Aprendizaje del Ecualizador (Ntaps={ntaps_test})')
    ax_err.set_xlabel('Iteraciones (Símbolos)')
    ax_err.set_ylabel(r'Error Cuadrático Medio ($10\log_{10}|e|^2$)')
    ax_err.grid(True, linestyle='--')
    ax_err.legend()
    fig3.tight_layout()

    plt.show()
