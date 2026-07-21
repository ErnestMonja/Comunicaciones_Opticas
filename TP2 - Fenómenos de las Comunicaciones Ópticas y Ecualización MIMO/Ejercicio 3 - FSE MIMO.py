import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fft as fft
from scipy.interpolate import interp1d
from types import SimpleNamespace
import time

# Funciones auxiliares
import m_channel as mc
from communications import *

# ---------------------------------------------------------
# PARAMETROS
# ---------------------------------------------------------

# Parametros TX
Lsim = int(300e3) # Symbols
M = 16            # QPSK=4, QAM16=16
BR = 32e9        # Symbol rate

# Parametros generales
NRX = 2 # Receiver oversampling ratio
NOS = 2 # Channel oversampling ratio

# Parametros del canal: solo se utiliza la rotacion del SOP
# snr_db = np.array([10])
snr_db = np.linspace(10, 20, 11) # Array de SNR [dB]
Lfiber = 0 # km (1000)
bcde_Lfiber = 0 # Esta surge de un algoritmo de estimacion de long de fibra (1000)
DGD_ps = 0
SOPMD_ps2 = 60 * DGD_ps
PCD_ps2 = 30 * DGD_ps
# fSOP_tx = np.array([100e3])
fSOP_tx = np.array([0, 30e3, 60e3, 100e3])

# Parametros del receptor
Ntaps = 31
# step = np.array([6e-3])
step = np.linspace(2e-3, 6e-3, 3)
target_agc = 0.3

# Plots
en_plots = False

# Estructura
# TX --> Pulse Shaping --> Canal PMD --> Ruido --> Antialias + decim
# (opcional) --> AGC --> FFE MIMO --> decim --> BER

# -----------------------------------------------------------------------------
# TRANSMISOR
# -----------------------------------------------------------------------------
pulse_shaping_type = 0
# Pulse shaping con RRC-RC
if pulse_shaping_type == 0:
    type_shape = 'sqrt'
else:
    type_shape = 'normal'

htx = root_raised_cosine(NOS, 0.1, 15)

# TX H
dec_labels_h = np.random.randint(0, M, Lsim)
tx_symbs_h = qammod(dec_labels_h, M)
xup_h = np.zeros(len(tx_symbs_h) * NOS, dtype=np.complex128)
xup_h[::NOS] = tx_symbs_h
sh = sig.lfilter(htx, [1.0], xup_h)
RCMA = np.sqrt(np.mean(np.abs(tx_symbs_h)**4) / np.mean(np.abs(tx_symbs_h)**2))

# TX V
dec_labels_v = np.random.randint(0, M, Lsim)
tx_symbs_v = qammod(dec_labels_v, M)
xup_v = np.zeros(len(tx_symbs_v) * NOS, dtype=np.complex128)
xup_v[::NOS] = tx_symbs_v
sv = sig.lfilter(htx, [1.0], xup_v)

# -----------------------------------------------------------------------------
# MEDICION DE BER: para un SNR, f_SOP y step
# -----------------------------------------------------------------------------
# Tiempo de ejecucion de la funcion: 67 segundos
def ber_measure(snr_db_v, fSOP_tx, step):

    # -----------------------------------------------------------------------------
    # CANAL
    # -----------------------------------------------------------------------------
    # Calculo de OSNR para entrar a la funcion m_channel
    osnr_db = snr_db_v + 10*np.log10(BR / 12.5e9)

    config_channel = SimpleNamespace()
    channel_input = SimpleNamespace()

    # Load inputs
    channel_input.h_signal_v = sh
    channel_input.v_signal_v = sv

    # Load config
    config_channel.OVR = NOS                  # Oversampling factor
    config_channel.BR = BR                    # Symbol Rate [Bd]
    config_channel.CD_D_ps_nm_km = 20
    config_channel.lambda_m = 1550e-9
    config_channel.link_len_m = Lfiber * 1000 # Fiber length [m]

    config_channel.DGD_ps = DGD_ps            # Differential Group Delay [ps]
    config_channel.SOPMD_ps2 = SOPMD_ps2      # Polarization Mode Dispersion [ps^2]
    config_channel.SOPMD_CD_ps2 = PCD_ps2     # Chromatic Dispersion due PMD [ps^2]

    config_channel.fSOP_tx = fSOP_tx          # Tx Rot of the States of Pol [Hz]
    config_channel.fSOP_rx = 0e3              # Rx Rot of the States of Pol [Hz]

    config_channel.osnr_db = osnr_db          # Channel SNR [dB] --> PENDING
    config_channel.en_noise = 1               # 0:OFF | 1:ON

    config_channel.en_ideal_fiber = 0         # 0:OFF | 1:ON

    o_data_s, _ = mc.m_channel(channel_input, config_channel)
    ych_h = o_data_s.h_signal_v
    ych_v = o_data_s.v_signal_v

    # Agregar lo offset y ruido de fase (en este caso no se utiliza)
    lo_offset = 0e6
    LW = 0e3
    Ls = len(ych_h)
    nline = np.arange(Ls)
    tline = nline / (NOS * BR)
    laser_lo_offset = np.exp(1j * tline * 2 * np.pi * lo_offset)
    wnoise = np.sqrt(2 * np.pi * LW / (NOS * BR)) * np.random.randn(Ls)
    laser_pnoise = np.exp(1j * np.cumsum(wnoise))
    ych_h = ych_h * laser_lo_offset * laser_pnoise
    ych_v = ych_v * laser_lo_offset * laser_pnoise

    # -----------------------------------------------------------------------------
    # ANTIALIAS (OPCIONAL) solo si hacemos cambio de tasa
    # -----------------------------------------------------------------------------
    y_aaf_h = ych_h
    y_aaf_v = ych_v

    # -----------------------------------------------------------------------------
    # AGC
    # -----------------------------------------------------------------------------
    metric_h = np.std(y_aaf_h, ddof=1)
    metric_v = np.std(y_aaf_v, ddof=1)
    y_agc_h = y_aaf_h / metric_h * target_agc
    y_agc_v = y_aaf_v / metric_v * target_agc

    # -----------------------------------------------------------------------------
    # Ecualizador BCDE
    # -----------------------------------------------------------------------------
    if bcde_Lfiber > 0:
        lambda_nm = 1550      # 1550nm
        Dfiber = 0.02         # 20ps/nm/km
        fftsize = 128 * 1024  # Arranco con una FFT grande x las dudas

        # vector de frecuencias angulares
        w = 2*np.pi * np.fft.fftfreq(fftsize, d=1/(BR*NRX))

        beta2 = lambda_nm**2 * Dfiber / (2*np.pi*3e8)

        # respuesta en frecuencia de CD
        bcde_freq_resp = np.exp(1j * beta2 / 2 * bcde_Lfiber * w**2 ) # H(w)
        
        # Recortamos la respuesta para reducir complejidad
        bcde_imp_resp_full = fft.fftshift(fft.ifft(bcde_freq_resp))
        accum_energy = np.cumsum(np.abs(bcde_imp_resp_full)**2)
        normalized_energy = accum_energy / accum_energy[-1]
        
        mm = np.where((normalized_energy > (1 - 99.99 / 100)) & (normalized_energy < (99.99 / 100)))[0]
        bcde_imp_resp_trim = bcde_imp_resp_full[mm]
        required_ntaps_bcde = len(bcde_imp_resp_trim)
        
        # Usar filtrado en el dominio de la frecuencia es lo mejor
        tentative_nfft = 2 * required_ntaps_bcde
        nfft_bcde = 2 ** int(np.ceil(np.log2(tentative_nfft)))
        overlap = nfft_bcde // 2 
        rta_impulso = np.concatenate([bcde_imp_resp_trim, np.zeros(nfft_bcde - len(bcde_imp_resp_trim))])
        
        rta_freq = fft.fft(rta_impulso)
        block_size = nfft_bcde - overlap
        Ls_blocks = (len(y_agc_h) // block_size) * block_size
        nblocks = Ls_blocks // block_size
        
        y_bcde_h = np.zeros(Ls_blocks, dtype=complex)
        y_bcde_v = np.zeros(Ls_blocks, dtype=complex)
        
        for nb in range(1, nblocks):
            start_in = nb * block_size - overlap
            end_in = (nb + 1) * block_size
            slice_input = slice(start_in, end_in)
            
            fft_input_h = y_agc_h[slice_input]
            fft_input_v = y_agc_v[slice_input]
            
            fft_output_h = fft.fft(fft_input_h)
            fft_output_v = fft.fft(fft_input_v)
            
            filter_out_h = fft_output_h * rta_freq
            filter_out_v = fft_output_v * rta_freq
            
            ifft_output_h = fft.ifft(filter_out_h)
            ifft_output_v = fft.ifft(filter_out_v)
            
            start_out = nb * block_size
            end_out = (nb + 1) * block_size
            y_bcde_h[start_out:end_out] = ifft_output_h[overlap:]
    else:
        y_bcde_h = y_agc_h
        y_bcde_v = y_agc_v

    # -----------------------------------------------------------------------------
    # Ecualizacion MIMO
    # -----------------------------------------------------------------------------
    Ls = len(y_bcde_h)
    eq_buffer_h = np.zeros(Ntaps, dtype=complex)
    eq_buffer_v = np.zeros(Ntaps, dtype=complex)
    heq_00 = np.zeros(Ntaps, dtype=complex)
    heq_01 = np.zeros(Ntaps, dtype=complex)
    heq_10 = np.zeros(Ntaps, dtype=complex)
    heq_11 = np.zeros(Ntaps, dtype=complex)

    heq_00[Ntaps // 2] = 1
    heq_11[Ntaps // 2] = 1

    yh_log = np.zeros(Ls, dtype=complex)
    yv_log = np.zeros(Ls, dtype=complex)
    log_len = int(np.ceil(Ls / NRX)) + 2

    error_h_log = np.zeros(log_len, dtype=complex)
    error_v_log = np.zeros(log_len, dtype=complex)
    ffe_output_h_log = np.zeros(log_len, dtype=complex)
    ffe_output_v_log = np.zeros(log_len, dtype=complex)
    dec_h_log = np.zeros(log_len, dtype=complex)
    dec_v_log = np.zeros(log_len, dtype=complex)
    fcr_output_h_log = np.zeros(log_len, dtype=complex)
    fcr_output_v_log = np.zeros(log_len, dtype=complex)
    fcr_nco = np.zeros((log_len, 2))
    fcr_inte = np.zeros((log_len, 2))

    # kp_fcr = 25e-3
    kp_fcr = 0
    ki_fcr = kp_fcr / 1000
    fcr_latency = 0

    subsf = int(1e3 / 2)
    log_subsf_len = int(np.ceil((Ls / NRX) / subsf)) + 2
    h00_log = np.zeros((log_subsf_len, Ntaps), dtype=complex)
    h01_log = np.zeros((log_subsf_len, Ntaps), dtype=complex)
    h10_log = np.zeros((log_subsf_len, Ntaps), dtype=complex)
    h11_log = np.zeros((log_subsf_len, Ntaps), dtype=complex)

    # SM
    enable_cma = 1  # 0 es DD, 1 es CMA
    enable_fcr = 0 
    normal_operation = 0 # declarar que esta todo ok

    timer_cma = int(np.ceil(0.3 * Lsim))
    timer_fcr = int(np.ceil(0.3 * Lsim))
    timer_dd = int(np.ceil(0.1 * Lsim))
    normal_op_timer = timer_cma + timer_fcr + timer_dd
    force_cma = False

    # time_start = time.perf_counter()
    for n in range(2 * fcr_latency + 2, Ls):
        if n == Ls/4:
            print('25%...')
        if n == Ls/2:
            print('50%...')
        if n == Ls*3/4:
            print('75%...')

        # Meto nueva muestra en el buffer
        eq_buffer_h[1:] = eq_buffer_h[:-1]
        eq_buffer_h[0] = y_bcde_h[n]
        eq_buffer_v[1:] = eq_buffer_v[:-1]
        eq_buffer_v[0] = y_bcde_v[n]
        
        # Producto MIMO
        yffe_h = np.sum(eq_buffer_h * heq_00) + np.sum(eq_buffer_v * heq_01)
        yffe_v = np.sum(eq_buffer_h * heq_10) + np.sum(eq_buffer_v * heq_11)
        yh_log[n] = yffe_h
        yv_log[n] = yffe_v
        
        if (n + 1) % NRX == 0:
            n2 = (n + 1) // NRX - 1
            n3 = n2 // subsf
            
            if n2 < timer_cma:
                enable_cma = 1
                enable_fcr = 0 
                normal_operation = 0
            elif n2 < (timer_cma + timer_dd):
                # enable_cma = 1
                enable_cma = 0
                enable_fcr = 1 
                normal_operation = 0
            elif n2 < (timer_fcr + timer_cma + timer_dd):
                enable_cma = 0
                enable_fcr = 1 
                normal_operation = 0
            else:
                enable_cma = 0
                enable_fcr = 1 
                normal_operation = 1
                
            ffe_output_h_log[n2] = yffe_h
            ffe_output_v_log[n2] = yffe_v
            
            yfcr_h = yffe_h * np.exp(-1j * fcr_nco[n2 - fcr_latency, 0])
            yfcr_v = yffe_v * np.exp(-1j * fcr_nco[n2 - fcr_latency, 1])
            fcr_output_h_log[n2] = yfcr_h
            fcr_output_v_log[n2] = yfcr_v
            
            if not enable_cma or enable_fcr:
                dec_h = slicer_QAM(yfcr_h, M)
                dec_v = slicer_QAM(yfcr_v, M)
                    
            if enable_cma or force_cma:
                error_cma_h = yffe_h * (np.abs(yffe_h) - RCMA)
                error_cma_v = yffe_v * (np.abs(yffe_v) - RCMA)
                error_h_log[n2] = error_cma_h
                error_v_log[n2] = error_cma_v
                error_h = error_cma_h
                error_v = error_cma_v
            else:
                dec_h_log[n2] = dec_h
                dec_v_log[n2] = dec_v
                error_dd_bb_h = yfcr_h - dec_h # Base band
                error_dd_bb_v = yfcr_v - dec_v
                error_h_log[n2] = error_dd_bb_h
                error_v_log[n2] = error_dd_bb_v
                error_dd_pb_h = error_dd_bb_h * np.exp(1j * fcr_nco[n2 - fcr_latency, 0]) # pass band error
                error_dd_pb_v = error_dd_bb_v * np.exp(1j * fcr_nco[n2 - fcr_latency, 1])
                error_h = error_dd_pb_h
                error_v = error_dd_pb_v
                
            if enable_fcr:
                yfcr = [yfcr_h, yfcr_v]
                dec = [dec_h, dec_v]
                for pol in range(2):
                    phase_error = -np.angle(dec[pol]) + np.angle(yfcr[pol])
                    prop_error = kp_fcr * phase_error
                    fcr_inte[n2 + 1, pol] = fcr_inte[n2, pol] + ki_fcr * phase_error
                    fcr_nco[n2 + 1, pol] = fcr_nco[n2, pol] + fcr_inte[n2 + 1, pol] + prop_error
                
            heq_00 = heq_00 * (1 - step * 1e-2) - step * np.conj(eq_buffer_h) * error_h
            heq_01 = heq_01 * (1 - step * 1e-2) - step * np.conj(eq_buffer_v) * error_h
            heq_10 = heq_10 * (1 - step * 1e-2) - step * np.conj(eq_buffer_h) * error_v
            heq_11 = heq_11 * (1 - step * 1e-2) - step * np.conj(eq_buffer_v) * error_v
            
            if (n2 + 1) % subsf == 0:
                h00_log[n3, :] = heq_00
                h01_log[n3, :] = heq_01
                h10_log[n3, :] = heq_10
                h11_log[n3, :] = heq_11

    # time_stop = time.perf_counter()
    # print(f'Tiempo de ejecucion del FSE: {time_stop - time_start} segundos.')

    # Trimming excess from pre-allocation
    Ls_rx = int(np.ceil(Ls / NRX))
    ffe_output_h_log = ffe_output_h_log[:Ls_rx]
    ffe_output_v_log = ffe_output_v_log[:Ls_rx]
    fcr_output_h_log = fcr_output_h_log[:Ls_rx]
    fcr_output_v_log = fcr_output_v_log[:Ls_rx]

    # -----------------------------------------------------------------------------
    # PLOTS
    # -----------------------------------------------------------------------------
    # Plot constellations
    if en_plots:
        valid = slice(int(0.8 * Lsim), int(0.9 * Lsim))
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(np.real(fcr_output_h_log[valid]), np.imag(fcr_output_h_log[valid]), '.')
        plt.grid(True)
        plt.axis('equal')
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        plt.xlabel("Real Part")
        plt.ylabel("Imag Part")
        plt.title("Constellation H")

        plt.subplot(1, 2, 2)
        plt.plot(np.real(fcr_output_v_log[valid]), np.imag(fcr_output_v_log[valid]), '.')
        plt.grid(True)
        plt.axis('equal')
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        plt.xlabel("Real Part")
        plt.ylabel("Imag Part")
        plt.title("Constellation V")

        # Evolution of symbols
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(np.real(fcr_output_h_log), '.')
        plt.xlabel('Samples'); plt.ylabel('Amplitude'); plt.title('HI'); plt.grid(True)
        plt.subplot(2, 2, 2)
        plt.plot(np.imag(fcr_output_h_log), '.')
        plt.xlabel('Samples'); plt.ylabel('Amplitude'); plt.title('HQ'); plt.grid(True)
        plt.subplot(2, 2, 3)
        plt.plot(np.real(fcr_output_v_log), '.')
        plt.xlabel('Samples'); plt.ylabel('Amplitude'); plt.title('VI'); plt.grid(True)
        plt.subplot(2, 2, 4)
        plt.plot(np.imag(fcr_output_v_log), '.')
        plt.xlabel('Samples'); plt.ylabel('Amplitude'); plt.title('VQ'); plt.grid(True)

        # Rta final FFE
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(np.real(heq_00), '-ob', label='Real')
        plt.plot(np.imag(heq_00), '-or', label='Imag')
        plt.xlabel('Samples'); plt.ylabel('Imp. Resp.'); plt.title('H00'); plt.grid(True); plt.ylim([-3, 3])
        plt.subplot(2, 2, 2)
        plt.plot(np.real(heq_01), '-ob')
        plt.plot(np.imag(heq_01), '-or')
        plt.xlabel('Samples'); plt.ylabel('Imp. Resp.'); plt.title('H01'); plt.grid(True); plt.ylim([-3, 3])
        plt.subplot(2, 2, 3)
        plt.plot(np.real(heq_10), '-ob')
        plt.plot(np.imag(heq_10), '-or')
        plt.xlabel('Samples'); plt.ylabel('Imp. Resp.'); plt.title('H10'); plt.grid(True); plt.ylim([-3, 3])
        plt.subplot(2, 2, 4)
        plt.plot(np.real(heq_11), '-ob')
        plt.plot(np.imag(heq_11), '-or')
        plt.xlabel('Samples'); plt.ylabel('Imp. Resp.'); plt.title('H11'); plt.grid(True); plt.ylim([-3, 3])

        h00_log = h00_log[:(Ls_rx // subsf)]
        h01_log = h01_log[:(Ls_rx // subsf)]
        h10_log = h10_log[:(Ls_rx // subsf)]
        h11_log = h11_log[:(Ls_rx // subsf)]
        nline_sub = subsf * np.arange(len(h00_log))

        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(nline_sub, np.abs(h00_log))
        plt.xlabel('Symbols'); plt.ylabel('Abs. Value'); plt.title('HH'); plt.grid(True)
        plt.subplot(2, 2, 2)
        plt.plot(nline_sub, np.abs(h01_log))
        plt.xlabel('Symbols'); plt.ylabel('Abs. Value'); plt.title('HV'); plt.grid(True)
        plt.subplot(2, 2, 3)
        plt.plot(nline_sub, np.abs(h10_log))
        plt.xlabel('Symbols'); plt.ylabel('Abs. Value'); plt.title('VH'); plt.grid(True)
        plt.subplot(2, 2, 4)
        plt.plot(nline_sub, np.abs(h11_log))
        plt.xlabel('Symbols'); plt.ylabel('Abs. Value'); plt.title('VV'); plt.grid(True)

        plt.figure(figsize=(10, 8))
        nfft_plt = 1024
        fs_plt = NRX * BR / 1e9
        dF = fs_plt / nfft_plt
        fv = np.arange(0, nfft_plt) * dF

        plt.subplot(2, 2, 1)
        plt.plot(fv, 20 * np.log10(np.abs(fft.fft(heq_00, nfft_plt))))
        plt.grid(True); plt.xlabel('Freq [GHz]'); plt.ylabel('Mag. Freq. Resp.'); plt.title('HH')
        plt.subplot(2, 2, 2)
        plt.plot(fv, 20 * np.log10(np.abs(fft.fft(heq_01, nfft_plt))))
        plt.grid(True); plt.xlabel('Freq [GHz]'); plt.ylabel('Mag. Freq. Resp.'); plt.title('HV')
        plt.subplot(2, 2, 3)
        plt.plot(fv, 20 * np.log10(np.abs(fft.fft(heq_10, nfft_plt))))
        plt.grid(True); plt.xlabel('Freq [GHz]'); plt.ylabel('Mag. Freq. Resp.'); plt.title('VH')
        plt.subplot(2, 2, 4)
        plt.plot(fv, 20 * np.log10(np.abs(fft.fft(heq_11, nfft_plt))))
        plt.grid(True); plt.xlabel('Freq [GHz]'); plt.ylabel('Mag. Freq. Resp.'); plt.title('VV')

    # -----------------------------------------------------------------------------
    # Calculo de BER
    # -----------------------------------------------------------------------------
    # Differential correlator
    Ltrim = int(np.ceil(Lsim / 2)) - 1
    ffe_output_trim = np.zeros((len(fcr_output_h_log) - Ltrim - int(1e3), 2), dtype=complex)
    ffe_output_trim[:, 0] = fcr_output_h_log[Ltrim : -int(1e3)]
    ffe_output_trim[:, 1] = fcr_output_v_log[Ltrim : -int(1e3)]

    tx_trim = np.zeros((len(tx_symbs_h) - Ltrim - int(1e3), 2), dtype=complex)
    tx_trim[:, 0] = tx_symbs_h[Ltrim : -int(1e3)]
    tx_trim[:, 1] = tx_symbs_v[Ltrim : -int(1e3)]

    Lcorr = 20000
    rx_diff = np.zeros((Lcorr - 1, 2), dtype=complex)
    tx_diff = np.zeros((Lcorr - 1, 2), dtype=complex)

    rx_diff[:, 0] = ffe_output_trim[1:Lcorr, 0] * np.conj(ffe_output_trim[0:Lcorr-1, 0])
    tx_diff[:, 0] = tx_trim[1:Lcorr, 0] * np.conj(tx_trim[0:Lcorr-1, 0])
    rx_diff[:, 1] = ffe_output_trim[1:Lcorr, 1] * np.conj(ffe_output_trim[0:Lcorr-1, 1])
    tx_diff[:, 1] = tx_trim[1:Lcorr, 1] * np.conj(tx_trim[0:Lcorr-1, 1])

    # Este correlador no puede arreglar reflex (cambios de signo por lane en TX)
    corrs = [[None, None], [None, None]]
    best_nrx = [0, 0]

    if en_plots:
        plt.figure(figsize=(10, 8))

    for ntx in range(2):
        max_corr = 0
        best_nrx[ntx] = 0
        for nrx in range(2):
            
            if en_plots:
                plt.subplot(2, 2, ntx * 2 + nrx + 1)
                
            # Using full xcorr analog logic
            corr_val = np.abs(sig.correlate(tx_diff[:, ntx], rx_diff[:, nrx], mode='full'))
            normalized_corr = corr_val / Lcorr / np.var(tx_symbs_h[1:Lcorr], ddof=1)
            corrs[ntx][nrx] = normalized_corr

            if en_plots:
                plt.plot(normalized_corr)
                plt.grid(True)
                
            if np.max(normalized_corr) > max_corr:
                max_corr = np.max(normalized_corr)
                best_nrx[ntx] = nrx

    if best_nrx == [0, 1]:
        swap = False
        colision = False
    elif best_nrx == [1, 0]:
        swap = True
        colision = False
    else:
        swap = False
        colision = True

    # assert not colision, "Colisión en el enrutamiento cruzado."
    if colision:
        print("Colision en el enrutamiento cruzado.")
    if swap:
        print("Swap en el enrutamiento cruzado.")

    delay_h = finddelay(tx_diff[:, 0], rx_diff[:, best_nrx[0]])
    delay_v = finddelay(tx_diff[:, 1], rx_diff[:, best_nrx[1]])

    rx_align = np.zeros((len(ffe_output_trim) - max(delay_h, delay_v, 0), 2), dtype=complex)
    tx_align = np.zeros_like(rx_align)

    rx_align[:, 0] = ffe_output_trim[max(delay_h, 0) : max(delay_h, 0) + len(rx_align), 0]
    rx_align[:, 1] = ffe_output_trim[max(delay_v, 0) : max(delay_v, 0) + len(rx_align), 1]

    tx_align[:, 0] = tx_trim[max(-delay_h, 0) : max(-delay_h, 0) + len(rx_align), best_nrx[0]]
    tx_align[:, 1] = tx_trim[max(-delay_v, 0) : max(-delay_v, 0) + len(rx_align), best_nrx[1]]

    # En este punto tengo las seniales alineadas
    # Ejecuto el CSC dinamico
    csc_block_size = 128
    Nblocks = len(rx_align) // csc_block_size
    rx_fix_csc = np.zeros((Nblocks * csc_block_size, 2), dtype=complex)

    for pol in range(2):
        for nb in range(Nblocks):
            start_idx = nb * csc_block_size
            end_idx = (nb + 1) * csc_block_size
            csc_rx_in = rx_align[start_idx:end_idx, pol]
            csc_tx_in = tx_align[start_idx:end_idx, pol]
            
            best_p = 0
            best_mse = np.inf
            for ptest in [0, np.pi/2, np.pi, 1.5 * np.pi]:
                mse = np.mean(np.abs(csc_tx_in - csc_rx_in * np.exp(1j * ptest))**2)
                if mse < best_mse:
                    best_mse = mse
                    best_p = ptest
            
            rx_fix_csc[start_idx:end_idx, pol] = csc_rx_in * np.exp(1j * best_p)

    tx_fix_csc = tx_align[:Nblocks * csc_block_size, :]

    # Calculo de BER
    ber_per_pol = np.zeros(2)
    for pol in range(2):
        rx_decs = slicer_QAM(rx_fix_csc[:, pol], M)
        errors_symb = np.sum(rx_decs != tx_fix_csc[:, pol])
        if errors_symb >= 100:
            ser = errors_symb / len(rx_decs)
            ber_per_pol[pol] = (1 / np.log2(M)) * ser
        else:
            ber_per_pol[pol] = 0
            break

    ber = np.mean(ber_per_pol)

    if en_plots:
        plt.show()

    return ber

def obtener_penalidad(fSOP, step):
    ber_sim = []
    ber_theo = []

    for snr_db_val in snr_db:
        print('------------------------------------------------------------------')
        print(f'Simulando: SNR = {snr_db_val} dB, fSOP = {fSOP} Hz, step = {step}')

        ber_sim_val = ber_measure(snr_db_val, fSOP, step)
        ber_theo_val = ber_theoretical(snr_db_val)

        ber_sim.append(ber_sim_val)
        ber_theo.append(ber_theo_val)
        print(f'BER medida = {ber_sim_val}\n, BER teorica = {ber_theo_val}')
    
    ber_sim = np.array(ber_sim)
    ber_theo = np.array(ber_theo)
    valid_idx = ber_sim > 0
    ber_sim_valid = ber_sim[valid_idx]
    snr_db_valid = snr_db[valid_idx]

    # Plots para comprobar la curva de BER
    # plt.figure()
    # plt.semilogy(snr_db, ber_sim, marker='o', label='sim')
    # plt.semilogy(snr_db, ber_theo, marker='o', label='theo')
    # plt.show()
    
    if len(ber_sim_valid) < 2 or np.min(ber_sim_valid) > 0.05:
        return 15.0
        
    b_log = np.log10(ber_sim_valid)
    idx_sort = np.argsort(b_log)
    b_log_s = b_log[idx_sort]
    e_rng_s = snr_db_valid[idx_sort]
    
    _, unique_idx = np.unique(b_log_s, return_index=True)
    b_log_u = b_log_s[unique_idx]
    e_rng_u = e_rng_s[unique_idx]
    
    if len(b_log_u) < 2:
        return 15.0
        
    try:
        f_sim = interp1d(b_log_u, e_rng_u, kind='linear', fill_value="extrapolate")
        snr_sim_1e3 = f_sim(-3.0) 
    except Exception:
        return 15.0
    
    ber_theo_log = np.log10(ber_theo + 1e-12)
    f_teo = interp1d(ber_theo_log, snr_db, kind='linear', fill_value="extrapolate")
    snr_theo_1e3 = f_teo(-3.0)
    
    penalty = snr_sim_1e3 - snr_theo_1e3
    # return penalty
    return max(0.0, min(penalty, 15.0))



# -----------------------------------------------------------------------------
# FUNCION MAIN
# -----------------------------------------------------------------------------
plt.figure()

for step_val in step:
    snr_penalty = []
    for fSOP in fSOP_tx:
        snr_penalty_val = obtener_penalidad(fSOP, step_val)
        snr_penalty.append(snr_penalty_val)
        print(f'SNR penalty, fSOP = {fSOP} -> {snr_penalty_val} dB')

    snr_penalty = np.array(snr_penalty)

    plt.plot(fSOP_tx/1e3, snr_penalty, marker='o', label=rf'$\mu = ${step_val:.3f}')

plt.title('Penalidad de SNR debido a la rotación del SOP')
plt.xlabel(r'$f_{SOP}$ [kHz]')
plt.ylabel(r'Penalidad de SNR [dB] @ BER = $10^{-3}$')
plt.grid()
plt.legend()
plt.show()