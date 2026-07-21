import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import group_delay
from scipy.interpolate import interp1d
from types import SimpleNamespace

def get_cd_coeffitient_itu(lambda_nm_v, type_val=0):
    """
    Interpolates Chromatic Dispersion (CD) coefficients for ITU G.652.D fibers.
    """
    # Maximum value CD coefficient (ITU G.652.D fibers)
    # cd_coeff_max_m[:, 0] is lambda in nm
    # cd_coeff_max_m[:, 1] is D in ps / (nm * km)
    cd_coeff_max_m = np.array([
        [1273.0404, -1.9085], [1282.1480, -1.1330], [1291.1576, -0.3654], [1300.1673, 0.4001],
        [1309.2749, 1.1706], [1318.3825, 1.9365], [1327.4900, 2.6993], [1336.7935, 3.4717],
        [1346.1949, 4.2463], [1355.5962, 5.0126], [1365.0955, 5.7793], [1374.6928, 6.5453],
        [1384.3879, 7.3083], [1394.2790, 8.0777], [1404.3659, 8.8495], [1414.5507, 9.6170],
        [1424.8334, 10.3774], [1435.4100, 11.1462], [1446.1824, 11.9151], [1457.1506, 12.6831],
        [1468.4127, 13.4596], [1479.7727, 14.2306], [1491.3286, 15.0028], [1503.1782, 15.7850],
        [1515.2237, 16.5699], [1527.3672, 17.3525], [1539.4127, 18.1227], [1551.5561, 18.8878],
        [1563.7975, 19.6585], [1576.1368, 20.4277], [1588.6719, 21.2080], [1601.1092, 21.9775],
        [1613.5464, 22.7468], [1622.6540, 23.3112]
    ])

    # Minimum value CD coefficient (ITU G.652.D fibers)
    cd_coeff_min_m = np.array([
        [1273.8239, -4.8284], [1282.4418, -4.0513], [1290.9618, -3.2862], [1299.5797, -2.5186],
        [1308.2956, -1.7534], [1317.2073, -0.9813], [1326.3149, -0.2081], [1335.5204, 0.5575],
        [1345.0197, 1.3278], [1354.9107, 2.1080], [1364.8997, 2.8701], [1375.2804, 3.6385],
        [1386.1507, 4.4154], [1397.2169, 5.1749], [1408.8707, 5.9449], [1421.3079, 6.7321],
        [1434.1369, 7.5111], [1447.2596, 8.2767], [1460.9700, 9.0502], [1475.3658, 9.8377],
        [1490.1534, 10.6236], [1505.2348, 11.4055], [1520.7079, 12.1912], [1536.4747, 12.9762],
        [1552.4375, 13.7600], [1568.3023, 14.5283], [1584.3630, 15.2980], [1600.8154, 16.0819],
        [1617.2678, 16.8604]
    ])

    min_lambda = 1250
    max_lambda = 1610

    if np.min(lambda_nm_v) < min_lambda or np.max(lambda_nm_v) > max_lambda:
        raise ValueError(f"Lambda values must be between {min_lambda:.1f} and {max_lambda:.1f} [nm]")

    # Interpolation
    f_max = interp1d(cd_coeff_max_m[:, 0], cd_coeff_max_m[:, 1], kind='cubic', fill_value='extrapolate')
    f_min = interp1d(cd_coeff_min_m[:, 0], cd_coeff_min_m[:, 1], kind='cubic', fill_value='extrapolate')

    cd_coef_D_max = f_max(lambda_nm_v)
    cd_coef_D_min = f_min(lambda_nm_v)

    if type_val == 0:
        return cd_coef_D_max
    elif type_val == 1:
        return cd_coef_D_min
    elif type_val == 2: # WORST
        idx_max_v = np.abs(cd_coef_D_max) > np.abs(cd_coef_D_min)
        return np.where(idx_max_v, cd_coef_D_max, cd_coef_D_min)
    elif type_val == 3: # BEST
        idx_max_v = np.abs(cd_coef_D_max) < np.abs(cd_coef_D_min)
        return np.where(idx_max_v, cd_coef_D_max, cd_coef_D_min)
    else:
        raise ValueError("Unknown CD D type.-")

def raised_cosine(fc, fs, rolloff, n_taps, t0=0):
    rolloff = rolloff + 0.0001
    Ts = 1 / fs
    T = 1 / fc
    
    if n_taps % 2 == 0:
        n_taps += 1
        
    t_v = np.arange(-(n_taps - 1) / 2, (n_taps - 1) / 2 + 1) * Ts + t0
    tn_v = t_v * 2 / T
    
    h_v = np.sinc(tn_v) * (np.cos(np.pi * rolloff * tn_v)) / (1 - (2 * rolloff * tn_v) ** 2)
    return h_v / np.sum(h_v)

def plot_cwdm4_D_and_lanes():
    lambda_nm_v = np.arange(1260, 1343)
    
    x_lane_0 = [1261, 1281]
    x_lane_1 = [1281, 1301]
    x_lane_2 = [1301, 1321]
    x_lane_3 = [1321, 1341]
    
    y_up_lanes = [20, 20]
    basevalue = -10
    y_max = np.min(np.array(y_up_lanes) + basevalue) * -1
    y_min = np.min(np.array(y_up_lanes) + basevalue)
    
    D_up = get_cd_coeffitient_itu(lambda_nm_v, 0)
    D_down = get_cd_coeffitient_itu(lambda_nm_v, 1)
    D_worst = get_cd_coeffitient_itu(lambda_nm_v, 2)
    D_better = get_cd_coeffitient_itu(lambda_nm_v, 3)

    plt.figure(figsize=(8, 7), facecolor='w')
    alpha = 0.25

    plt.fill_between(x_lane_0, basevalue, y_up_lanes[0], color='b', alpha=alpha, label='Lane 0')
    plt.fill_between(x_lane_1, basevalue, y_up_lanes[0], color='g', alpha=alpha, label='Lane 1')
    plt.fill_between(x_lane_2, basevalue, y_up_lanes[0], color='y', alpha=alpha, label='Lane 2')
    plt.fill_between(x_lane_3, basevalue, y_up_lanes[0], color='r', alpha=alpha, label='Lane 3')

    plt.plot(lambda_nm_v, D_up, '-r', linewidth=5, label='D up')
    plt.plot(lambda_nm_v, D_down, '-b', linewidth=5, label='D down')
    plt.plot(lambda_nm_v, D_worst, '--m', linewidth=5, label='$|D|$ max')
    plt.plot(lambda_nm_v, D_better, '--g', linewidth=5, label='$|D|$ min')

    plt.xlim([min(lambda_nm_v), max(lambda_nm_v)])
    plt.ylim([y_min, y_max])
    plt.title("ITU-T G.652 type fiber CD coefficient\nIMDD CWDM4", fontsize=14)
    plt.xlabel(r'$\lambda$ [$nm$]', fontsize=14)
    plt.ylabel(r'D [ps/(nm $\cdot$ km)]', fontsize=14)
    plt.grid(True)
    plt.legend(loc='upper left', ncol=2, fontsize=12)
    plt.show()

def single_mode_fiber(**kwargs):
    # DEFAULT SETTINGS
    cfg_s = SimpleNamespace(
        enable_plots=True,
        fs=4 * 200e9,
        nfft=2**13,
        lambda_m=1329e-9,
        link_m=2e3,
        slice_energy_per=99.9 / 100,
        CD_D_type=2,
        CD_D_ps_nm_km=0,
        CD_S_ps_nm2_km=0.092,
        DGD_type=0,
        DGD_ps=10,
        PMDq_ps_sqrt_km=0.5,
        PMDn=3.75,
        SOPMD_type=0,
        SOPMD_ps2=20,
        SOPMD_CD_ps2=0,
        measure_PMD=True,
        h_tx_v=np.nan,
        h_rx_v=np.nan
    )
    
    # Overwrite configuration with user inputs
    for k, v in kwargs.items():
        if not hasattr(cfg_s, k):
            raise ValueError(f"{k}: Invalid parameter.")
        setattr(cfg_s, k, v)

    # Local assignment of parameters for cleaner formulas
    fs = cfg_s.fs
    nfft = int(cfg_s.nfft)
    lambda_m = cfg_s.lambda_m
    link_m = cfg_s.link_m
    slice_energy_per = cfg_s.slice_energy_per
    CD_D_type = cfg_s.CD_D_type
    CD_D_ps_nm_km = cfg_s.CD_D_ps_nm_km
    CD_S_ps_nm2_km = cfg_s.CD_S_ps_nm2_km
    DGD_type = cfg_s.DGD_type
    DGD_ps = cfg_s.DGD_ps
    PMDn = cfg_s.PMDn
    PMDq_ps_sqrt_km = cfg_s.PMDq_ps_sqrt_km
    SOPMD_type = cfg_s.SOPMD_type
    SOPMD_ps2 = cfg_s.SOPMD_ps2
    SOPMD_CD_ps2 = cfg_s.SOPMD_CD_ps2
    measure_PMD = cfg_s.measure_PMD
    enable_plots = cfg_s.enable_plots
    
    # DEPENDANT VARIABLES
    fs_ghz = fs / 1e9
    nfft2 = nfft // 2
    
    delta_f = fs_ghz / nfft
    delta_w = 2 * np.pi * delta_f

    w_pos_v = np.arange(0, nfft2) * delta_w
    w_neg_v = np.arange(-nfft2, 0) * delta_w
    w_grad_sec_v = np.concatenate([w_pos_v, w_neg_v])

    # ELECTRICAL RESPONSE
    if np.any(np.isnan(cfg_s.h_tx_v)):
        h_tx_v = raised_cosine(0.8 * fs / 2, fs, 0.1, 101, 0)
        cfg_s.h_tx_v = h_tx_v
    else:
        h_tx_v = cfg_s.h_tx_v
        
    if np.any(np.isnan(cfg_s.h_rx_v)):
        h_rx_v = np.array([1.0])
        cfg_s.h_rx_v = h_rx_v
    else:
        h_rx_v = cfg_s.h_rx_v

    h_txrx_v = np.convolve(h_tx_v, h_rx_v)
    
    # Calculate group delay
    _, n_delay = group_delay((h_txrx_v, 1.0), w=1)
    tau = n_delay[0] / fs_ghz
    H_txrx_v = np.fft.fft(h_txrx_v, nfft)

    # CHROMATIC DISPERSION (CD)
    if CD_D_type != 4:
        lambda_nm = lambda_m / 1e-9
        CD_D_ps_nm_km = get_cd_coeffitient_itu(lambda_nm, CD_D_type)
        cfg_s.CD_D_ps_nm_km = CD_D_ps_nm_km

    c_m_sec = 3e8
    CD_D_sec_m2 = CD_D_ps_nm_km * 1e-12 / (1e-9 * 1e3)
    CD_S_sec_m3 = CD_S_ps_nm2_km * 1e-12 / ((1e-9)**2 * 1e3)
    w_rad_sec_v = w_grad_sec_v * 1e9

    beta2 = -CD_D_sec_m2 * lambda_m**2 / (2 * np.pi * c_m_sec)
    
    beta3 = 0
    if not np.isnan(CD_S_ps_nm2_km):
        beta3 = (CD_S_sec_m3 * lambda_m**4 / (2 * np.pi * c_m_sec)**2 + 
                 CD_D_sec_m2 * lambda_m**3 / (2 * np.pi**2 * c_m_sec**2))
      
    beta = 1/2 * beta2 * w_rad_sec_v**2 + 1/6 * beta3 * w_rad_sec_v**3
    H_cd_v = np.exp(-1j * beta * link_m)

    # POLARIZATION MODE DISPERSION (PMD)
    if DGD_type != 1:
        link_km = link_m / 1e3
        DGD_ps = PMDn * PMDq_ps_sqrt_km * np.sqrt(link_km)
        cfg_s.DGD_ps = DGD_ps
    
    if SOPMD_type != 1:
        SOPMD_ps2 = np.sqrt(15 * DGD_ps)
        SOPMD_CD_ps2 = np.sqrt(15 * DGD_ps)
        cfg_s.SOPMD_ps2 = SOPMD_ps2
        cfg_s.SOPMD_CD_ps2 = SOPMD_CD_ps2

    DGD_ns = DGD_ps * 1e-3
    SOPMD_ns2 = SOPMD_ps2 * (1e-3)**2
    SOPMD_CD_ns2 = SOPMD_CD_ps2 * (1e-3)**2

    pmd_cd_ns2 = SOPMD_CD_ns2
    if SOPMD_ns2 < SOPMD_CD_ns2:
        pmd_cd_ns2 = SOPMD_ns2 / np.sqrt(2)
    
    k_ns = 0
    pdm_dep_ns2 = np.sqrt(SOPMD_ns2**2 - pmd_cd_ns2**2)
    if DGD_ns > 0:
        k_ns = pdm_dep_ns2 / DGD_ns / 4

    kw_v = k_ns * w_grad_sec_v

    H_pmd_v = np.exp(1j * DGD_ns * w_grad_sec_v / 2 + 1j * pmd_cd_ns2 * w_grad_sec_v**2 / 4)

    # R(w)
    R_11_v = + np.cos(kw_v)
    R_12_v = + np.sin(kw_v)
    R_21_v = - np.sin(kw_v)
    R_22_v = + np.cos(kw_v)

    # D(w)
    D_11_v = H_pmd_v
    D_12_v = np.zeros(nfft)
    D_21_v = np.zeros(nfft)
    D_22_v = np.conj(H_pmd_v)

    # X(w) = D(w) * R(w)
    X_11_v = D_11_v * R_11_v + D_12_v * R_21_v
    X_12_v = D_11_v * R_12_v + D_12_v * R_22_v
    X_21_v = D_21_v * R_11_v + D_22_v * R_21_v
    X_22_v = D_21_v * R_12_v + D_22_v * R_22_v

    # R(w)^(-1)
    Ri_11_v = + np.cos(kw_v)
    Ri_12_v = - np.sin(kw_v)
    Ri_21_v = + np.sin(kw_v)
    Ri_22_v = + np.cos(kw_v)

    # M(w) = R(w)^(-1) * X(w)
    M_11_v = Ri_11_v * X_11_v + Ri_12_v * X_21_v
    M_12_v = Ri_11_v * X_12_v + Ri_12_v * X_22_v
    M_21_v = Ri_21_v * X_11_v + Ri_22_v * X_21_v
    M_22_v = Ri_21_v * X_12_v + Ri_22_v * X_22_v

    # Measure DGD and SOPMD
    if measure_PMD:
        dM_11_v = (np.roll(M_11_v, -1) - M_11_v) / delta_w
        dM_12_v = (np.roll(M_12_v, -1) - M_12_v) / delta_w
        dM_21_v = (np.roll(M_21_v, -1) - M_21_v) / delta_w
        dM_22_v = (np.roll(M_22_v, -1) - M_22_v) / delta_w

        P_11_v = -2j * (dM_11_v * np.conj(M_11_v) + dM_12_v * np.conj(M_12_v))
        P_21_v = -2j * (dM_21_v * np.conj(M_11_v) + dM_22_v * np.conj(M_12_v))
        
        omega = np.array([np.real(P_21_v[0]), np.imag(P_21_v[0]), np.real(P_11_v[0])])
        domega = np.array([
            np.real(P_21_v[1]) - np.real(P_21_v[0]),
            np.imag(P_21_v[1]) - np.imag(P_21_v[0]),
            np.real(P_11_v[1]) - np.real(P_11_v[0])
        ]) / delta_w

        cfg_s.meas_DGD_ps = 1e3 * np.linalg.norm(omega)
        cfg_s.meas_SOPMD_ps2 = (1e3)**2 * np.linalg.norm(domega)

    # CHANNEL FREQUENCY RESPONSE
    CH_11_v = M_11_v * H_cd_v * H_txrx_v
    CH_12_v = M_12_v * H_cd_v * H_txrx_v
    CH_21_v = M_21_v * H_cd_v * H_txrx_v
    CH_22_v = M_22_v * H_cd_v * H_txrx_v

    # CHANNEL IMPULSE RESPONSE
    h_11_long_v = np.fft.fftshift(np.fft.ifft(CH_11_v, nfft))
    h_12_long_v = np.fft.fftshift(np.fft.ifft(CH_12_v, nfft))
    h_21_long_v = np.fft.fftshift(np.fft.ifft(CH_21_v, nfft))
    h_22_long_v = np.fft.fftshift(np.fft.ifft(CH_22_v, nfft))

    # Cut the pulse response with the 99% of energy
    h_cum_v = np.cumsum((np.abs(h_11_long_v) + np.abs(h_12_long_v) + 
                         np.abs(h_21_long_v) + np.abs(h_22_long_v))**2)
    
    if slice_energy_per > 1 or slice_energy_per < 0:
        raise ValueError('Parameter "slice_energy_per" must be between 0 and 1.-')
                
    h_pow_max = h_cum_v[-1]
    up_limit = h_pow_max * slice_energy_per
    dw_limit = h_pow_max * (1 - slice_energy_per)
    
    i_start = np.argmax(h_cum_v >= dw_limit) - 1
    i_end = len(h_cum_v) - np.argmax(h_cum_v[::-1] <= up_limit) - 1
    
    idx_v = np.arange(i_start + 1, i_end + 2)

    h_11_v = h_11_long_v[idx_v]
    h_12_v = h_12_long_v[idx_v]
    h_21_v = h_21_long_v[idx_v]
    h_22_v = h_22_long_v[idx_v]

    # PLOT
    if enable_plots:
        plot_cwdm4_D_and_lanes()
        
        n_taps = len(h_11_v)
        n_v = np.arange(1, n_taps + 1)
        f_v = np.fft.fftshift(w_grad_sec_v) / (2 * np.pi)
        f0_i = np.where(f_v == 0)[0][0]
        
        fig, axs = plt.subplots(4, 2, figsize=(14, 16), facecolor='w')
        axs = axs.flatten()

        def plot_impulse_and_phase(idx, h_v, h_long_v, CH_v, title_t, title_f):
            # Time domain plot
            ax_t = axs[idx*2]
            ax_t.plot(n_v, np.real(h_v), '-or', label='Real')
            ax_t.plot(n_v, np.imag(h_v), '-ob', label='Imag')
            ax_t.grid(True)
            ax_t.set_xlim([0, n_taps])
            ax_t.set_title(title_t, fontsize=14)
            ax_t.set_xlabel('Samples', fontsize=12)
            ax_t.set_ylabel('Amplitude', fontsize=12)
            ax_t.legend(loc='lower right')

            # Frequency domain plot
            ax_f = axs[idx*2 + 1]
            h_aux_v = np.zeros(nfft, dtype=complex)
            h_aux_v[idx_v] = h_long_v[idx_v]
            
            y_filter_v = np.unwrap(np.fft.fftshift(np.angle(np.fft.fft(np.fft.fftshift(h_aux_v), nfft))))
            y_filter_v -= y_filter_v[f0_i]
            
            y_theo_v = np.unwrap(np.fft.fftshift(np.angle(CH_v)))
            y_theo_v -= y_theo_v[f0_i]
            
            ax_f.plot(f_v, y_filter_v, '-r', linewidth=2, label='Measured')
            ax_f.plot(f_v, y_theo_v, '--k', linewidth=2, label='Theoretical')
            ax_f.grid(True)
            ax_f.set_title(title_f, fontsize=14)
            ax_f.set_xlabel('Frequency [GHz]', fontsize=12)
            ax_f.set_ylabel('Phase [rad]', fontsize=12)
            ax_f.legend(loc='upper center')

        plot_impulse_and_phase(0, h_11_v, h_11_long_v, CH_11_v, r'$h_{11}(t)$', r'$\angle H_{11}(\omega)$')
        plot_impulse_and_phase(1, h_12_v, h_12_long_v, CH_12_v, r'$h_{12}(t)$', r'$\angle H_{12}(\omega)$')
        plot_impulse_and_phase(2, h_21_v, h_21_long_v, CH_21_v, r'$h_{21}(t)$', r'$\angle H_{21}(\omega)$')
        plot_impulse_and_phase(3, h_22_v, h_22_long_v, CH_22_v, r'$h_{22}(t)$', r'$\angle H_{22}(\omega)$')

        plt.tight_layout()
        plt.show()
        
    return h_11_v, h_12_v, h_21_v, h_22_v, cfg_s