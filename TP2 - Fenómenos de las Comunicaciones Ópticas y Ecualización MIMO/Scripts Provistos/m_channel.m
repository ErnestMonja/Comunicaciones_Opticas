%-----------------------------------------------------------------------------%
%                                   FULGOR
%
% Programmer(s): Santiago F. Leguizamon
% Created on   : July 2023
% Description  : Optical fiber channel model
%-----------------------------------------------------------------------------%

function [o_data_s, o_cfg_s] = m_channel(i_data_s, i_cfg_s)

    % // TODO: agregar ruido ASE
    
    %--------------------------%
    %     DEFAULT SETTINGS
    %--------------------------%

    cfg_s.OVR               = 2         ; % Oversampling factor
    cfg_s.BR                = 240e9     ; % Symbol Rate [Bd]
    
    cfg_s.nfft              = 2^15      ; % Fiber model NFFT
    
    cfg_s.lambda            = 1500e-9   ; % Wavelength [m]
    cfg_s.link_len_m        = 0         ; % Fiber length [m]
    cfg_s.CD_D_ps_nm_km     = 18.8      ; % CD D [ps/nm/km]
    cfg_s.CD_S_ps_nm2_km    = 0.09      ; % CD S [ps/nm^2/km]
    cfg_s.CD_D_type         = 4         ; % 0) Max D from ITU G.652.D fibers
                                          % 1) Min D from ITU 
                                          % 2) Worst D (max |D|) from ITU 
                                          % 3) Best D (min |D|) from ITU 
                                          % 4) User D in (ps / (nm * km))
    
    cfg_s.DGD_ps            = 0         ; % Differential Group Delay [ps]
    cfg_s.DGD_type          = 1         ; % 0: PMDn*PMDq*sqrt(L). 1: User DGD       
    cfg_s.PMDq_ps_sqrt_km   = 0.2       ; % PMDq [ps/sqrt(km)]
    cfg_s.PMDn              = 3.75      ; % PMDn [-]
    
    cfg_s.SOPMD_type        = 1         ; % 0: SOPMD=SOPMD_CD=sqrt(15*DGD_ps)
                                          % 1: User SOPMD and SOPMD_CD
    cfg_s.SOPMD_ps2         = 0         ; % Polarization Mode Dispersion [ps^2]
    cfg_s.SOPMD_CD_ps2      = 0         ; % Chromatic Dispersion due PMD [ps^2]
    
    cfg_s.htxd              = NaN       ; % Tx electrical response
    cfg_s.hrxd              = NaN       ; % Rx electrical response
    
    cfg_s.fSOP_tx           = 0e3       ; % Tx Rot of the States of Pol [Hz]
    cfg_s.fSOP_rx           = 0e3       ; % Rx Rot of the States of Pol [Hz]
    
    cfg_s.osnr_db           = 22        ; % Channel SNR [dB]
    cfg_s.en_noise          = 1         ; % 0:OFF | 1:ON
    
    cfg_s.bw                = 100e9     ; % Channel BW
    cfg_s.bw_order          = 6         ; % Channel filter order
    cfg_s.en_bw_lim         = 0         ; % 0:OFF | 1:ON
    
    cfg_s.en_ideal_fiber    = 0         ; % 0:OFF | 1:ON
    
    %--------------------------%
    %       REASSIGNMENT
    %--------------------------%

    if nargin > 1
        cfg_s = overwrite_parameters(i_cfg_s, cfg_s);
    end

    clear i_cfg_s;
    
    %--------------------------%
    %           ERRORS
    %--------------------------%
    
    %--------------------------%
    %   CONSTANTS & VARIABLES
    %--------------------------%
    
    fs_ch = cfg_s.OVR*cfg_s.BR;
    
    % BW limitation
    if cfg_s.en_bw_lim && cfg_s.hrxd==1
        
        % Filter design
        fcn = 2 * cfg_s.bw / (cfg_s.OVR*cfg_s.BR);
        h_ch_v = fir1(cfg_s.bw_order, fcn);

        cfg_s.hrxd = h_ch_v;

    end

    % Fiber model
    if cfg_s.en_ideal_fiber
        
        h11 = 1;
        h12 = 0;
        h21 = 0;
        h22 = 1;
        
    else
                                            
        [h11, h12, h21, h22, cfg_s.fiber_s] = single_mode_fiber( ...
                                'enable_plots', 0, ...
                                'fs', fs_ch, ...
                                'nfft', cfg_s.nfft, ...
                                'lambda_m', cfg_s.lambda, ...
                                'link_m', cfg_s.link_len_m, ...
                                'slice_energy_per', 99.99 / 100, ...
                                'CD_D_type', cfg_s.CD_D_type, ...
                                'CD_D_ps_nm_km', cfg_s.CD_D_ps_nm_km, ...
                                'CD_S_ps_nm2_km', cfg_s.CD_S_ps_nm2_km, ...
                                'DGD_type', cfg_s.DGD_type, ...
                                'DGD_ps', cfg_s.DGD_ps, ...
                                'PMDq_ps_sqrt_km', cfg_s.PMDq_ps_sqrt_km, ...
                                'PMDn', cfg_s.PMDn, ...
                                'SOPMD_type', cfg_s.SOPMD_type, ...
                                'SOPMD_ps2', cfg_s.SOPMD_ps2, ...
                                'SOPMD_CD_ps2', cfg_s.SOPMD_CD_ps2, ...
                                'measure_PMD', false, ...
                                'h_tx_v', cfg_s.htxd, ...
                                'h_rx_v', cfg_s.hrxd);
    
    end

    % Attenuation
    lambda_nm = cfg_s.lambda/1e-9;
    link_len_km = cfg_s.link_len_m / 1e3;
    attenuation_dB_per_km = 0;%get_fiber_attenuation_at_lambda(lambda_nm);
    att_times_per_km = 10^(attenuation_dB_per_km/10);
    
    % SNR
    snr_ch = 10^(cfg_s.osnr_db/10) * 12.5e9 / (cfg_s.OVR*cfg_s.BR);
    
    %--------------------------%
    %          PROCESS
    %--------------------------%
        
    % fSOP Tx
    if cfg_s.fSOP_tx > 0
        t_v = (0 : length(i_data_s.h_signal_v) - 1) .* 1 / (cfg_s.BR*cfg_s.OVR);
        titaSOP = 2 * pi * cfg_s.fSOP_tx .* t_v;
        o_sop_tx_h = cos(titaSOP)' .* i_data_s.h_signal_v - ...
                                        sin(titaSOP)' .* i_data_s.v_signal_v;
        o_sop_tx_v = cos(titaSOP)' .* i_data_s.v_signal_v + ...
                                        sin(titaSOP)' .* i_data_s.h_signal_v;
    else
        o_sop_tx_h = i_data_s.h_signal_v;
        o_sop_tx_v = i_data_s.v_signal_v;
    end
    
    % Filter through fiber response
    o_fiber_h = filter(h11, 1, o_sop_tx_h) + filter(h12, 1, o_sop_tx_v);
    o_fiber_v = filter(h22, 1, o_sop_tx_v) + filter(h21, 1, o_sop_tx_h);
    
    % fSOP Rx
    if cfg_s.fSOP_rx > 0
        t_v = (0 : length(i_data_s.h_signal_v) - 1) .* 1 / (cfg_s.BR*cfg_s.OVR);
        titaSOP = 2 * pi * cfg_s.fSOP_rx .* t_v;
        o_sop_h = cos(titaSOP)' .* o_fiber_h - sin(titaSOP)' .* o_fiber_v;
        o_sop_v = cos(titaSOP)' .* o_fiber_v + sin(titaSOP)' .* o_fiber_h;
    else
        o_sop_h = o_fiber_h;
        o_sop_v = o_fiber_v;
    end
    
    % Noise addition
    pw_noise = var(o_sop_h + o_sop_v) / snr_ch;
    
    if cfg_s.en_noise
        noise_h_v = sqrt(pw_noise/4) * ...
                            (randn(size(o_sop_h)) + 1j* randn(size(o_sop_h)));
        noise_v_v = sqrt(pw_noise/4) * ...
                            (randn(size(o_sop_v)) + 1j* randn(size(o_sop_v)));
    else
        noise_h_v = 0;
        noise_v_v = 0;
    end
    
    o_sop_h = o_sop_h + noise_h_v;
    o_sop_v = o_sop_v + noise_v_v;
    
    % Attenuation
    if link_len_km>0
        att = sqrt(att_times_per_km*link_len_km);
    else
        att = 1;
    end
    
    %--------------------------%
    %          OUTPUT
    %--------------------------%
    
    o_data_s.h_signal_v = att * o_sop_h;
    o_data_s.v_signal_v = att * o_sop_v;
    o_cfg_s = cfg_s;
    
    o_data_s.h11 = h11;
    o_data_s.h12 = h12;
    o_data_s.h21 = h21;
    o_data_s.h22 = h22;
    
    %--------------------------%
    %           PLOTS
    %--------------------------%

end