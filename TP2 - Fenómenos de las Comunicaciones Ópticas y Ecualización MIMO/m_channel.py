import numpy as np
from scipy.signal import lfilter, firwin
from types import SimpleNamespace

# Modelo de fibra
import single_mode_fiber as smf

def m_channel(i_data_s, i_cfg_s=None):
    """
    Optical fiber channel model.
    """
    
    #--------------------------#
    #     DEFAULT SETTINGS
    #--------------------------#
    
    cfg_s = SimpleNamespace(
        OVR=2,              # Oversampling factor
        BR=240e9,           # Symbol Rate [Bd]
        nfft=2**15,         # Fiber model NFFT
        lambda_m=1500e-9,   # Wavelength [m] (renamed from 'lambda' to avoid reserved keyword)
        link_len_m=0,       # Fiber length [m]
        CD_D_ps_nm_km=18.8, # CD D [ps/nm/km]
        CD_S_ps_nm2_km=0.09,# CD S [ps/nm^2/km]
        CD_D_type=4,        # 0) Max D from ITU G.652.D fibers
                            # 1) Min D from ITU 
                            # 2) Worst D (max |D|) from ITU 
                            # 3) Best D (min |D|) from ITU 
                            # 4) User D in (ps / (nm * km))
        DGD_ps=0,           # Differential Group Delay [ps]
        DGD_type=1,         # 0: PMDn*PMDq*sqrt(L). 1: User DGD
        PMDq_ps_sqrt_km=0.2,# PMDq [ps/sqrt(km)]
        PMDn=3.75,          # PMDn [-]
        SOPMD_type=1,       # 0: SOPMD=SOPMD_CD=sqrt(15*DGD_ps)
                            # 1: User SOPMD and SOPMD_CD
        SOPMD_ps2=0,        # Polarization Mode Dispersion [ps^2]
        SOPMD_CD_ps2=0,     # Chromatic Dispersion due PMD [ps^2]
        htxd=np.nan,        # Tx electrical response
        hrxd=np.nan,        # Rx electrical response
        fSOP_tx=0e3,        # Tx Rot of the States of Pol [Hz]
        fSOP_rx=0e3,        # Rx Rot of the States of Pol [Hz]
        osnr_db=22,         # Channel SNR [dB]
        en_noise=1,         # 0:OFF | 1:ON
        bw=100e9,           # Channel BW
        bw_order=6,         # Channel filter order
        en_bw_lim=0,        # 0:OFF | 1:ON
        en_ideal_fiber=0    # 0:OFF | 1:ON
    )
    
    #--------------------------#
    #       REASSIGNMENT
    #--------------------------#
    
    if i_cfg_s is not None:
        # Overwrite defaults with elements from i_cfg_s
        for key, value in vars(i_cfg_s).items():
            setattr(cfg_s, key, value)
            
    #--------------------------#
    #   CONSTANTS & VARIABLES
    #--------------------------#
    
    fs_ch = cfg_s.OVR * cfg_s.BR
    
    # BW limitation
    if cfg_s.en_bw_lim and (np.isscalar(cfg_s.hrxd) and cfg_s.hrxd == 1):
        # Filter design
        fcn = 2 * cfg_s.bw / (cfg_s.OVR * cfg_s.BR)
        # Note: MATLAB's fir1(N, fcn) creates an N-th order filter which has N+1 taps.
        h_ch_v = firwin(cfg_s.bw_order + 1, fcn, window='hamming')
        cfg_s.hrxd = h_ch_v

    # Fiber model
    if cfg_s.en_ideal_fiber:
        h11 = np.array([1.0])
        h12 = np.array([0.0])
        h21 = np.array([0.0])
        h22 = np.array([1.0])
    else:
        # Calls the function translated in the previous step
        h11, h12, h21, h22, cfg_s.fiber_s = smf.single_mode_fiber(
            enable_plots=False,
            fs=fs_ch,
            nfft=cfg_s.nfft,
            lambda_m=cfg_s.lambda_m,
            link_m=cfg_s.link_len_m,
            slice_energy_per=99.99 / 100,
            CD_D_type=cfg_s.CD_D_type,
            CD_D_ps_nm_km=cfg_s.CD_D_ps_nm_km,
            CD_S_ps_nm2_km=cfg_s.CD_S_ps_nm2_km,
            DGD_type=cfg_s.DGD_type,
            DGD_ps=cfg_s.DGD_ps,
            PMDq_ps_sqrt_km=cfg_s.PMDq_ps_sqrt_km,
            PMDn=cfg_s.PMDn,
            SOPMD_type=cfg_s.SOPMD_type,
            SOPMD_ps2=cfg_s.SOPMD_ps2,
            SOPMD_CD_ps2=cfg_s.SOPMD_CD_ps2,
            measure_PMD=False,
            h_tx_v=cfg_s.htxd,
            h_rx_v=cfg_s.hrxd
        )
        
    # Attenuation
    lambda_nm = cfg_s.lambda_m / 1e-9
    link_len_km = cfg_s.link_len_m / 1e3
    attenuation_dB_per_km = 0  # get_fiber_attenuation_at_lambda(lambda_nm)
    att_times_per_km = 10 ** (attenuation_dB_per_km / 10)
    
    # SNR
    snr_ch = 10 ** (cfg_s.osnr_db / 10) * 12.5e9 / (cfg_s.OVR * cfg_s.BR)
    
    #--------------------------#
    #          PROCESS
    #--------------------------#
    
    n_samples = len(i_data_s.h_signal_v)
    t_v = np.arange(n_samples) * (1 / (cfg_s.BR * cfg_s.OVR))
    
    # fSOP Tx
    if cfg_s.fSOP_tx > 0:
        titaSOP = 2 * np.pi * cfg_s.fSOP_tx * t_v
        o_sop_tx_h = np.cos(titaSOP) * i_data_s.h_signal_v - np.sin(titaSOP) * i_data_s.v_signal_v
        o_sop_tx_v = np.cos(titaSOP) * i_data_s.v_signal_v + np.sin(titaSOP) * i_data_s.h_signal_v
    else:
        o_sop_tx_h = i_data_s.h_signal_v
        o_sop_tx_v = i_data_s.v_signal_v
        
    # Filter through fiber response
    o_fiber_h = lfilter(h11, [1.0], o_sop_tx_h) + lfilter(h12, [1.0], o_sop_tx_v)
    o_fiber_v = lfilter(h22, [1.0], o_sop_tx_v) + lfilter(h21, [1.0], o_sop_tx_h)
    
    # fSOP Rx
    if cfg_s.fSOP_rx > 0:
        titaSOP = 2 * np.pi * cfg_s.fSOP_rx * t_v
        o_sop_h = np.cos(titaSOP) * o_fiber_h - np.sin(titaSOP) * o_fiber_v
        o_sop_v = np.cos(titaSOP) * o_fiber_v + np.sin(titaSOP) * o_fiber_h
    else:
        o_sop_h = o_fiber_h
        o_sop_v = o_fiber_v

    # Noise addition
    # Note: MATLAB's var() uses ddof=1 by default (N-1).
    pw_noise = np.var(o_sop_h + o_sop_v, ddof=1) / snr_ch
    
    if cfg_s.en_noise:
        noise_h_v = np.sqrt(pw_noise / 4) * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
        noise_v_v = np.sqrt(pw_noise / 4) * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
    else:
        noise_h_v = 0
        noise_v_v = 0
        
    o_sop_h = o_sop_h + noise_h_v
    o_sop_v = o_sop_v + noise_v_v
    
    # Attenuation
    if link_len_km > 0:
        att = np.sqrt(att_times_per_km * link_len_km)
    else:
        att = 1
        
    #--------------------------#
    #          OUTPUT
    #--------------------------#
    
    o_data_s = SimpleNamespace()
    o_data_s.h_signal_v = att * o_sop_h
    o_data_s.v_signal_v = att * o_sop_v
    
    o_data_s.h11 = h11
    o_data_s.h12 = h12
    o_data_s.h21 = h21
    o_data_s.h22 = h22
    
    o_cfg_s = cfg_s
    
    return o_data_s, o_cfg_s