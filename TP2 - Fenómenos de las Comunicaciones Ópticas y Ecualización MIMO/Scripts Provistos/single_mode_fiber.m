%-------------------------------------------------------------------------%
% Filename		: single_mode_fiber.m
% Programmer    : Patricio Reus Merlo
% Created on	: 08/09/2023
% Description 	: SMF model with CD, DGD and SOPMD (PCD and PSD)
%-------------------------------------------------------------------------%

function varargout = single_mode_fiber(varargin)

    %-----------------------------------------%
    %            DEFAULT SETTINGS
    %-----------------------------------------%
    
    cfg_s.enable_plots = true;
    cfg_s.fs = 4 * 200e9;
    cfg_s.nfft = 2^13;
    cfg_s.lambda_m = 1329e-9;
    cfg_s.link_m = 2e3;
    cfg_s.slice_energy_per = 99.9 / 100;

    % Chromatic dispersion D type: CD_D_type
    %
    %   0) Maximum D from ITU G.652.D fibers
    %   1) Minimum D from ITU G.652.D fibers
    %   2) Worst D (max |D|) from ITU G.652.D fibers
    %   3) Best D (min |D|) from ITU G.652.D fibers
    %   4) User D in (ps / (nm * km))

    cfg_s.CD_D_type = 2;
    cfg_s.CD_D_ps_nm_km = 0;
    cfg_s.CD_S_ps_nm2_km = 0.092;

    % Differential group delay type: DGD_type
    %
    %   0) DGD = PMDn * PMDq_ps_sqrt_km * sqrt(link_length_km)
    %   1) User DGD in ps
    
    cfg_s.DGD_type = 0;
    cfg_s.DGD_ps = 10;
    cfg_s.PMDq_ps_sqrt_km = 0.5;
    cfg_s.PMDn = 3.75;
    
    % Second order PMD type: SOPMD_type
    %
    %   0) SOPMD_ps2 = SOPMD_CD_ps2 = sqrt(15 * DGD_ps)
    %   1) User SOPMD_ps2 and SOPMD_CD_ps2 in ps^2
    
    cfg_s.SOPMD_type = 0;
    cfg_s.SOPMD_ps2 = 20;
    cfg_s.SOPMD_CD_ps2 = 0;
    cfg_s.measure_PMD = true;

    cfg_s.h_tx_v = nan;
    cfg_s.h_rx_v = nan;

    cfg_s = overwrite_and_export_fields_to_workspace(cfg_s, varargin);
    
    %-----------------------------------------%
    %           DEPENDANT VARIABLES
    %-----------------------------------------%

    % Frequency domine axis
    fs_ghz = fs / 1e9;
    nfft2 = nfft / 2;
    
    delta_f = fs_ghz / nfft;
    delta_w = 2 * pi * delta_f;

    w_pos_v = (0 : nfft2 - 1) .* delta_w; % Direct 2pi * [0, fn = fs/2]
    w_neg_v = (- nfft2 : - 1) .* delta_w; % Fliped 2pi * [-fn = fs/2, 0)
    w_grad_sec_v = [w_pos_v, w_neg_v];
    clear w_pos_v w_neg_v

    %-----------------------------------------%
    %           ELECTRICAL RESPONSE
    %-----------------------------------------%

    % TX default response:
    % LPF with fc = 0.8 * fn
    if isnan(cfg_s.h_tx_v)
        h_tx_v = raised_cosine(0.8 * fs / 2, fs, 0.1, 101, 0);
        cfg_s.h_tx_v = h_tx_v;
    end
    
    % RX default response:
    % Impulse
    if isnan(cfg_s.h_rx_v)
        h_rx_v = 1;
        cfg_s.h_rx_v = h_rx_v;
    end

    h_txrx_v = conv(h_tx_v, h_rx_v);
    n_delay = grpdelay(h_txrx_v).';
    tau = n_delay(1) / fs_ghz;
    H_txrx_v = fft(h_txrx_v, nfft); % .* exp(1j * tau * w_grad_sec_v);
    clear txrx_h_v h_tx_v h_rx_v

    %-----------------------------------------%
    %        CHROMATIC DISPERSION (CD)
    %-----------------------------------------%

    % Mathematical summary based on [1][2][3]
    %
    % After a length L of Standard Single-Mode Fiber (SSMF),
    % the resultant frequency-domain signal can be expressed by:
    %
    %       Eout(w) = Ein(w) H_CD(w)
    %
    %       H_CD(w) = e^(-1j beta(w) L)
    %
    % Where beta(w) is the fiber propagation function of  w.
    % It can be expanded in Taylor series arroun the central
    % angular frequency w0 [3]:
    %
    %       beta(w) =     beta0 (w - w0)^0
    %               +     beta1 (w - w0)^1
    %               + 1/2 beta2 (w - w0)^2 
    %               + 1/6 beta3 (w - w0)^3 + ...
    %
    %       betax =  d^(x) beta(w) / dw^(x) | w = w0
    %
    % Where:
    %       - beta0: phase
    %       - beta1: represents the group velocity vg.
    %                beta1 = 1 / vg
    %
    %       - beta2: represents the dispersion of the group velocity
    %                This phenomenon is known as the group-velocity
    %                dispersion (GVD), and beta2 is the GVD parameter.
    %                beta2 = d beta1 / dw = - D lambda^2 / (2 pi c)
    %
    %       - beta3: called the third-order dispersion (TOD) parameter.
    %                beta3 = d beta2 / dw
    %                beta3 = S lambda^4 / (2 pi c)^2 
    %                      + D lambda^3 / (2 pi^2 c^2)
    %
    % Where D is called the dispersion parameter and S = dD / dlambda is
    % the slope of the dispersion parameter D
    %
    % [1] L. Anet Neto et al., "Simple Estimation of Fiber Dispersion and
    %     Laser Chirp Parameters Using the Downhill Simplex Fitting
    %     Algorithm," in Journal of Lightwave Technology, Jan.15, 2013
    %
    % [2] G. Agrawal, Nonlinear Fiber Optics, 4th ed. New York: Academic
    %     Press, 2006, (Optics and Photonics).
    % [3] M. Stern, J. P. Heritage and E. W. Chase, "Grating compensation
    %     of third-order fiber dispersion," in IEEE Journal of Quantum
    %     Electronics, vol. 28, no. 12, pp. 2742-2748, Dec. 1992

    if CD_D_type ~= 4
        lambda_nm = lambda_m / 1e-9;
        CD_D_ps_nm_km = get_cd_coeffitient_itu(lambda_nm, CD_D_type);
        cfg_s.CD_D_ps_nm_km = CD_D_ps_nm_km;
    end

    % Change of units
    c_m_sec = 3e8;
    CD_D_sec_m2 = CD_D_ps_nm_km * 1e-12 / (1e-9 * 1e3);
    CD_S_sec_m3 = CD_S_ps_nm2_km * 1e-12 / ((1e-9).^2 * 1e3);
    w_rad_sec_v = w_grad_sec_v * 1e9;

    % beta2 = - D lambda^2 / (2 pi c)
    % beta2: [s^2/m/rad]
    % D: [s/m^2]
    % lambda: [m]
    % c: [m/s]

    beta2 = - CD_D_sec_m2 * lambda_m^2 / (2 * pi * c_m_sec);
    
    % beta3 = S lambda^4 / (2 pi c)^2 + lambda^3 D / (2 pi^2 c^2)
    % beta3: [s^3/(rad^2 m)]
    % S: [s/m^3]
    % D: [s/m^2]
    % lambda: [m]
    % c: [m/s]

    beta3 = 0;
    if ~isnan(CD_S_ps_nm2_km)
    beta3 = CD_S_sec_m3 * lambda_m^4 / (2 * pi * c_m_sec)^2 ...
          + CD_D_sec_m2 * lambda_m^3 / (2 * pi^2 * c_m_sec^2);
    end
      
    % beta(w) = 1/2 beta2 * w^2 + 1/6 beta3 w^3
    beta = 1 / 2 * beta2 * w_rad_sec_v.^2 ...
         + 1 / 6 * beta3 * w_rad_sec_v.^3;

    H_cd_v = exp(-1j * beta .* link_m);
    clear beta2 beta3 beta

    %-----------------------------------------%
    %   POLARIZATION MODE DISPERSION (PMD)
    %-----------------------------------------%

    % Mathematical summary based on [4][5][6][7]
    %
    % PMD effects model is based on the Bruyère’s model where the PMD
    % operator is descibed by the matrix M(w):
    %
    %      M(w) = R(w)^(-1) * D(w) * R(w)
    %
    % Where:
    %
    %      R(w) = |  cos(k * w)  ,  sin(k * w)   | 
    %             | -sin(k * w)  ,  cos(k * w)   |
    %
    % R(w)^(-1) = |  cos(k * w)  , -sin(k * w)   |
    %             |  sin(k * w)  ,  cos(k * w)   |
    %
    %      D(w) = | e^(-j * A(w)),     0         |
    %             |     0        , e^(+j * A(w)) |
    %
    %      A(w) = (tau_DGD * w + tau_PCD * w^2) / 2
    %
    % [4] Julien Poirier, Michel Gadonna, Laurent Dupont. PMD Effects in
    %     Fiber Optic Transmission Systems. Fiber and Integrated Optics,
    %     2008, 27 (6), pp.559-578.
    %
    % [5] Shtaif, M., Mecozzi, A., Turr, M., and Nagel, J. A. 2000.
    %     A compensator for the effects of high-order polarisation mode
    %     dispersion in optical fibers. IEEE PTL 12(4):434–436.
    %
    % [6] Kim, N.-Y., and Park, N. 2005. Independently tunable first and 
    %     second-order polarisation-mode dispersion emulator. Photonics
    %     Technology Letters, IEEE 17(3):576–578.
    %
    % [7] Orlandini, A., and Vincetti, L. 2003. Comparison of the Jones
    %     matrix analytical models applied to optical system affected by
    %     high-order PMD. Journal of Lightwave Technology 21(6):1456–1464.

    if DGD_type ~= 1
        link_km = link_m / 1e3;
        DGD_ps = PMDn * PMDq_ps_sqrt_km * sqrt(link_km);
        cfg_s.DGD_ps = DGD_ps;
    end
    
    if SOPMD_type ~= 1
        SOPMD_ps2 = sqrt(15 * DGD_ps);
        SOPMD_CD_ps2 = sqrt(15 * DGD_ps);
        
        cfg_s.SOPMD_ps2 = SOPMD_ps2;
        cfg_s.SOPMD_CD_ps2 = SOPMD_CD_ps2;
    end

    % Change of units
    DGD_ns = DGD_ps * 1e-3;
    SOPMD_ns2 = SOPMD_ps2 * (1e-3)^2;
    SOPMD_CD_ns2 = SOPMD_CD_ps2 * (1e-3)^2;

    % PCD
    pmd_cd_ns2 = SOPMD_CD_ns2;
    if SOPMD_ns2 < SOPMD_CD_ns2
        pmd_cd_ns2 = SOPMD_ns2 / sqrt(2);
    end
    
    % PSD
    k_ns = 0;
    pdm_dep_ns2 = sqrt(SOPMD_ns2.^2 - pmd_cd_ns2.^2);
    if DGD_ns > 0
        k_ns = pdm_dep_ns2 / DGD_ns / 4;

        % k = SOPMD / DGD / 4
        % Calculating (k_ns) which represents depolarization phenomena.
        % Because of this phenomena, the different frequency components
        % have different rotations in their polarization
    end
    kw_v = k_ns * w_grad_sec_v;

    % DGD + PCD
    H_pmd_v = exp(1j * DGD_ns .* w_grad_sec_v / 2 ...
                + 1j * pmd_cd_ns2 .* w_grad_sec_v.^2 / 4);

    % R(w)
    R_11_v = + cos(kw_v);
    R_12_v = + sin(kw_v);
    R_21_v = - sin(kw_v);
    R_22_v = + cos(kw_v);

    % D(w)
    D_11_v = H_pmd_v;
    D_12_v = zeros(1, nfft);
    D_21_v = zeros(1, nfft);
    D_22_v = conj(H_pmd_v);
    clear H_pmd_v

    % X(w) = D(w) * R(w)
    X_11_v = D_11_v .* R_11_v + D_12_v .* R_21_v;
    X_12_v = D_11_v .* R_12_v + D_12_v .* R_22_v;
    X_21_v = D_21_v .* R_11_v + D_22_v .* R_21_v;
    X_22_v = D_21_v .* R_12_v + D_22_v .* R_22_v;
    clear D_* R_*

    % R(w)^(-1)
    Ri_11_v = + cos(kw_v);
    Ri_12_v = - sin(kw_v);
    Ri_21_v = + sin(kw_v);
    Ri_22_v = + cos(kw_v);
    clear kw_v

    % M(w) = R(w)^(-1) * X(w) = R(w)^(-1) * D(w) * R(w)
    M_11_v = Ri_11_v .* X_11_v + Ri_12_v .* X_21_v;
    M_12_v = Ri_11_v .* X_12_v + Ri_12_v .* X_22_v;
    M_21_v = Ri_21_v .* X_11_v + Ri_22_v .* X_21_v;
    M_22_v = Ri_21_v .* X_12_v + Ri_22_v .* X_22_v;
    clear X_* Ri_*

    % Measure DGD and SOPMD
    if measure_PMD
        
        % Compute derivative of M(w): dM(w) / dw
        dM_11_v = (circshift(M_11_v, -1) - M_11_v) ./ delta_w;
        dM_12_v = (circshift(M_12_v, -1) - M_12_v) ./ delta_w;
        dM_21_v = (circshift(M_21_v, -1) - M_21_v) ./ delta_w;
        dM_22_v = (circshift(M_22_v, -1) - M_22_v) ./ delta_w;

        % P(w) = -2j * (dM(w) / dw) * Hermitian{M(w)}
        P_11_v = -2j * (dM_11_v .* conj(M_11_v) + dM_12_v .* conj(M_12_v));
        P_21_v = -2j * (dM_21_v .* conj(M_11_v) + dM_22_v .* conj(M_12_v));
        clear dM_*
        
        % Compute vector Omega (polarization vector in  Stokes space):
        omega = [real(P_21_v(1)), imag(P_21_v(1)), real(P_11_v(1))];
        domega = [real(P_21_v(2)) - real(P_21_v(1)), ...
                  imag(P_21_v(2)) - imag(P_21_v(1)), ...
                  real(P_11_v(2)) - real(P_11_v(1))] ./ delta_w;
        clear P_11_v P_21_v

        % Compute DGD and SOPMD:
        cfg_s.meas_DGD_ps = 1e3 * norm(omega);
        cfg_s.meas_SOPMD_ps2 = (1e3)^2 * norm(domega);
    end

    %-----------------------------------------%
    %       CHANNEL FREQUENCY RESPONSE
    %-----------------------------------------%

    CH_11_v = M_11_v .* H_cd_v .* H_txrx_v;
    CH_12_v = M_12_v .* H_cd_v .* H_txrx_v;
    CH_21_v = M_21_v .* H_cd_v .* H_txrx_v;
    CH_22_v = M_22_v .* H_cd_v .* H_txrx_v;
    clear M_* H_cd_v H_txrx_v

    %-----------------------------------------%
    %        CHANNEL IMPULSE RESPONSE
    %-----------------------------------------%

    h_11_long_v = fftshift(ifft(CH_11_v, nfft));
    h_12_long_v = fftshift(ifft(CH_12_v, nfft));
    h_21_long_v = fftshift(ifft(CH_21_v, nfft));
    h_22_long_v = fftshift(ifft(CH_22_v, nfft));
    if ~enable_plots; clear CH_*; end

    % Cut the pulse response with the 99% of energy 
    % to reduce the length of time domine filters

    h_cum_v = cumsum((abs(h_11_long_v) + abs(h_12_long_v) ...
                    + abs(h_21_long_v) + abs(h_22_long_v)).^2);
    
    if slice_energy_per > 1 || cfg_s.slice_energy_per < 0
        error('Parameter "slice_energy_per" must be between 0 and 1.-');
    end
                
    h_pow_max = h_cum_v(end);
    up_limit = h_pow_max .* slice_energy_per;
    dw_limit = h_pow_max .* (1 - slice_energy_per);
    
    i_start = find(h_cum_v >= dw_limit, 1, 'first') - 1;
    i_end = find(h_cum_v <= up_limit, 1, 'last') - 1;
    idx_v = 1 + (i_start : i_end);
    clear h_cum_v

    h_11_v = h_11_long_v(idx_v);
    h_12_v = h_12_long_v(idx_v);
    h_21_v = h_21_long_v(idx_v);
    h_22_v = h_22_long_v(idx_v);
    if ~enable_plots; clear *_long_v; end
    
    %-----------------------------------------%
    %                  OUTPUT
    %-----------------------------------------%

    varargout = cell(1, 5);
    varargout{1} =  h_11_v;
    varargout{2} =  h_12_v;
    varargout{3} =  h_21_v;
    varargout{4} =  h_22_v;
    varargout{5} =  cfg_s;
    
    %-----------------------------------------%
    %                  PLOT
    %-----------------------------------------%

    if enable_plots

        %-------------------------------------%
        %              D VS LAMBDA
        %-------------------------------------%

        plot_cwdm4_D_and_lanes();
        
        %-------------------------------------%
        %           CHANNEL RESPONSE
        %-------------------------------------%
        
        n_taps = length(h_11_v);
        n_v = 1 : n_taps;
        f_v = fftshift(w_grad_sec_v) / (2 * pi);
        f0_i = find(f_v == 0);
        
        figure('Color', 'w');
        
        nexttile
        plot(n_v, real(h_11_v), '-or', n_v, imag(h_11_v), '-ob');
        grid on; hold on;
        xlim([0, n_taps])
        title('$h_{11}(t)$','Interpreter','latex', 'FontSize', 14);
        xlabel('Samples','Interpreter','latex', 'FontSize', 14);
        ylabel('Amplitude','Interpreter','latex', 'FontSize', 14);
        legend({'Real', 'Imag'}, 'Interpreter','latex', ...
                                 'FontSize', 14, 'Location', 'se');    
        nexttile
        h_aux_v = zeros(1, nfft);
        h_aux_v(idx_v) = h_11_long_v(idx_v);
        y_filter_v = unwrap(fftshift(angle(fft(fftshift(h_aux_v), nfft))));
        y_filter_v = y_filter_v - y_filter_v(f0_i) ;
        y_theo_v = unwrap(fftshift(angle(CH_11_v)));
        y_theo_v = y_theo_v - y_theo_v(f0_i);
        
        plot(f_v, y_filter_v, '-r', f_v, y_theo_v, '--k', 'Linewidth', 2);
        grid on; hold on;
        title('$\angle H_{11}(\omega)$','Interpreter','latex', 'FontSize', 14);
        xlabel('Frequency [GHz]','Interpreter','latex', 'FontSize', 14);
        ylabel('Phase [rad]','Interpreter','latex', 'FontSize', 14);
        legend({'Measured', 'Theoretical'}, 'Interpreter','latex', ...
                                 'FontSize', 14, 'Location', 'n');
        nexttile
        plot(n_v, real(h_12_v), '-or', n_v, imag(h_12_v), '-ob');
        grid on; hold on;
        xlim([0, n_taps])
        title('$h_{12}(t)$','Interpreter','latex', 'FontSize', 14);
        xlabel('Samples','Interpreter','latex', 'FontSize', 14);
        ylabel('Amplitude','Interpreter','latex', 'FontSize', 14);
        legend({'Real', 'Imag'}, 'Interpreter','latex', ...
                                 'FontSize', 14, 'Location', 'se');
        nexttile
        h_aux_v = zeros(1, nfft);
        h_aux_v(idx_v) = h_12_long_v(idx_v);
        y_filter_v = unwrap(fftshift(angle(fft(fftshift(h_aux_v), nfft))));
        y_filter_v = y_filter_v - y_filter_v(f0_i);
        y_theo_v = unwrap(fftshift(angle(CH_12_v)));
        y_theo_v = y_theo_v - y_theo_v(f0_i);
        
        plot(f_v, y_filter_v, '-r', f_v, y_theo_v, '--k', 'Linewidth', 2);
        grid on; hold on;
        title('$\angle H_{12}(\omega)$','Interpreter','latex', 'FontSize', 14);
        xlabel('Frequency [GHz]','Interpreter','latex', 'FontSize', 14);
        ylabel('Phase [rad]','Interpreter','latex', 'FontSize', 14);
        legend({'Measured', 'Theoretical'}, 'Interpreter','latex', ...
                                 'FontSize', 14, 'Location', 'n');    
        nexttile
        plot(n_v, real(h_21_v), '-or', n_v, imag(h_21_v), '-ob');
        grid on; hold on;
        xlim([0, n_taps])
        title('$h_{21}(t)$','Interpreter','latex', 'FontSize', 14);
        xlabel('Samples','Interpreter','latex', 'FontSize', 14);
        ylabel('Amplitude','Interpreter','latex', 'FontSize', 14);
        legend({'Real', 'Imag'}, 'Interpreter','latex', ...
                                 'FontSize', 14, 'Location', 'se');
        nexttile
        h_aux_v = zeros(1, nfft);
        h_aux_v(idx_v) = h_21_long_v(idx_v);
        y_filter_v = unwrap(fftshift(angle(fft(fftshift(h_aux_v), nfft))));
        y_filter_v = y_filter_v - y_filter_v(f0_i);
        y_theo_v = unwrap(fftshift(angle(CH_21_v)));
        y_theo_v = y_theo_v - y_theo_v(f0_i);
        
        plot(f_v, y_filter_v, '-r', f_v, y_theo_v, '--k', 'Linewidth', 2);
        grid on; hold on;
        title('$\angle H_{21}(\omega)$','Interpreter','latex', 'FontSize', 14);
        xlabel('Frequency [GHz]','Interpreter','latex', 'FontSize', 14);
        ylabel('Phase [rad]','Interpreter','latex', 'FontSize', 14);
        legend({'Measured', 'Theoretical'}, 'Interpreter','latex', ...
                                 'FontSize', 14, 'Location', 'n');
        nexttile
        plot(n_v, real(h_22_v), '-or', n_v, imag(h_22_v), '-ob');
        grid on; hold on;
        xlim([0, n_taps])
        title('$h_{22}(t)$','Interpreter','latex', 'FontSize', 14);
        xlabel('Samples','Interpreter','latex', 'FontSize', 14);
        ylabel('Amplitude','Interpreter','latex', 'FontSize', 14);
        legend({'Real', 'Imag'}, 'Interpreter','latex', ...
                                 'FontSize', 14, 'Location', 'se');
        nexttile
        h_aux_v = zeros(1, nfft);
        h_aux_v(idx_v) = h_22_long_v(idx_v);
        y_filter_v = unwrap(fftshift(angle(fft(fftshift(h_aux_v), nfft))));
        y_filter_v = y_filter_v - y_filter_v(f0_i);
        y_theo_v = unwrap(fftshift(angle(CH_22_v)));
        y_theo_v = y_theo_v - y_theo_v(f0_i);
        
        plot(f_v, y_filter_v, '-r', f_v, y_theo_v, '--k', 'Linewidth', 2);
        grid on; hold on;
        title('$\angle H_{22}(\omega)$','Interpreter','latex', 'FontSize', 14);
        xlabel('Frequency [GHz]','Interpreter','latex', 'FontSize', 14);
        ylabel('Phase [rad]','Interpreter','latex', 'FontSize', 14);
        legend({'Measured', 'Theoretical'}, 'Interpreter','latex', ...
                                 'FontSize', 14, 'Location', 'n');
    end
    
end

function cd_coef_D_v = get_cd_coeffitient_itu(lambda_nm_v, type)

    % Default
    if nargin < 2
        % Maxixmum D from ITU
        type = 0; 
    end

    %-----------------------------------------%
    % Maximum value CD coeffitient (ITU G.652.D fibers)
    % cd_coeff_max_m(:, 1) is lambda in nm
    % cd_coeff_max_m(:, 2) is D in ps / (nm * km)
    %-----------------------------------------%

    cd_coeff_max_m = [  1273.0404 , -1.9085 ; ...
                        1282.1480 , -1.1330 ; ...
                        1291.1576 , -0.3654 ; ...
                        1300.1673 , 0.4001  ; ...
                        1309.2749 , 1.1706  ; ...
                        1318.3825 , 1.9365  ; ...
                        1327.4900 , 2.6993  ; ...
                        1336.7935 , 3.4717  ; ...
                        1346.1949 , 4.2463  ; ...
                        1355.5962 , 5.0126  ; ...
                        1365.0955 , 5.7793  ; ...
                        1374.6928 , 6.5453  ; ...
                        1384.3879 , 7.3083  ; ...
                        1394.2790 , 8.0777  ; ...
                        1404.3659 , 8.8495  ; ...
                        1414.5507 , 9.6170  ; ...
                        1424.8334 , 10.3774 ; ...
                        1435.4100 , 11.1462 ; ...
                        1446.1824 , 11.9151 ; ...
                        1457.1506 , 12.6831 ; ...
                        1468.4127 , 13.4596 ; ...
                        1479.7727 , 14.2306 ; ...
                        1491.3286 , 15.0028 ; ...
                        1503.1782 , 15.7850 ; ...
                        1515.2237 , 16.5699 ; ...
                        1527.3672 , 17.3525 ; ...
                        1539.4127 , 18.1227 ; ...
                        1551.5561 , 18.8878 ; ...
                        1563.7975 , 19.6585 ; ...
                        1576.1368 , 20.4277 ; ...
                        1588.6719 , 21.2080 ; ...
                        1601.1092 , 21.9775 ; ...
                        1613.5464 , 22.7468 ; ...
                        1622.6540 , 23.3112 ];

    %-----------------------------------------%
    % Minimum value CD coeffitient (ITU G.652.D fibers)
    % cd_coeff_min_m(:, 1) is lambda in nm
    % cd_coeff_min_m(:, 2) is D in ps / (nm * km)
    %-----------------------------------------%

    cd_coeff_min_m = [  1273.8239 , -4.8284 ; ...
                        1282.4418 , -4.0513 ; ...
                        1290.9618 , -3.2862 ; ...
                        1299.5797 , -2.5186 ; ...
                        1308.2956 , -1.7534 ; ...
                        1317.2073 , -0.9813 ; ...
                        1326.3149 , -0.2081 ; ...
                        1335.5204 , 0.5575  ; ...
                        1345.0197 , 1.3278  ; ...
                        1354.9107 , 2.1080  ; ...
                        1364.8997 , 2.8701  ; ...
                        1375.2804 , 3.6385  ; ...
                        1386.1507 , 4.4154  ; ...
                        1397.2169 , 5.1749  ; ...
                        1408.8707 , 5.9449  ; ...
                        1421.3079 , 6.7321  ; ...
                        1434.1369 , 7.5111  ; ...
                        1447.2596 , 8.2767  ; ...
                        1460.9700 , 9.0502  ; ...
                        1475.3658 , 9.8377  ; ...
                        1490.1534 , 10.6236 ; ...
                        1505.2348 , 11.4055 ; ...
                        1520.7079 , 12.1912 ; ...
                        1536.4747 , 12.9762 ; ...
                        1552.4375 , 13.7600 ; ...
                        1568.3023 , 14.5283 ; ...
                        1584.3630 , 15.2980 ; ...
                        1600.8154 , 16.0819 ; ...
                        1617.2678 , 16.8604 ];
                    
    % Check measurement limits
    min_lambda = 1250;
    max_lambda = 1610;
    
    if (min(lambda_nm_v) < min_lambda) ...
        || (max(lambda_nm_v) > max_lambda)
        
        error("lambda values must be between %.1f and %.1f [nm]", ...
               min_lambda, max_lambda);
    end

    % Interpolation
    lambda_nm_itu_v = cd_coeff_max_m(:, 1);
    cd_coef_D_itu_v = cd_coeff_max_m(:, 2);

    cd_coef_D_max = interp1(lambda_nm_itu_v, cd_coef_D_itu_v, ...
                            lambda_nm_v, 'spline', 'extrap');
                        
    lambda_nm_itu_v = cd_coeff_min_m(:, 1);
    cd_coef_D_itu_v = cd_coeff_min_m(:, 2);
            
    cd_coef_D_min = interp1(lambda_nm_itu_v, cd_coef_D_itu_v, ...
                            lambda_nm_v, 'spline', 'extrap');
    
    switch type
        
        % MAX
        case 0
            cd_coef_D_v = cd_coef_D_max;

        % MIN    
        case 1
            cd_coef_D_v = cd_coef_D_min;
              
        % WORST
        case 2
            idx_max_v = abs(cd_coef_D_max) > abs(cd_coef_D_min);
            idx_min_v = abs(cd_coef_D_min) > abs(cd_coef_D_max);
            idx_v = [idx_min_v; idx_max_v];
            aux_m = [cd_coef_D_min; cd_coef_D_max];
            cd_coef_D_v = aux_m(idx_v);
            
        % BEST
        case 3
            idx_max_v = abs(cd_coef_D_max) < abs(cd_coef_D_min);
            idx_min_v = abs(cd_coef_D_min) < abs(cd_coef_D_max);
            idx_v = [idx_min_v; idx_max_v];
            aux_m = [cd_coef_D_min; cd_coef_D_max];
            cd_coef_D_v = aux_m(idx_v);
            
        otherwise
            error("Unknow CD D type.-")     
    end
end

function new_s = overwrite_and_export_fields_to_workspace(old_s, change)
    
    % Force to struct
    if iscell(change)
        change = struct(change{:});
    end

    % Overwrite parameters
    new_s = old_s;
    fn = fieldnames(change);
    for k = 1 : numel(fn)
        if ~isfield(new_s, fn{k})
            error("%s: Invalid parameter.", fn{k});
        end
        new_s.(fn{k}) = change.(fn{k});
    end

    % From struct to workspace of caller function
    fn = fieldnames(new_s);
    for i = 1 : numel(fn)
        assignin('caller', fn{i}, new_s.(fn{i}));
    end
end

function [h_v, n_taps] = raised_cosine(fc, fs, rolloff, n_taps, t0)
    
    if nargin < 5
        t0 = 0;
    end

    rolloff = rolloff + 0.0001; 
    Ts = 1 / fs ;
    T  = 1 / fc ;
    
    % Force to odd
    aux= n_taps;
    if mod(aux,2)
        n_taps=aux+1;
    end
    
    % Time vector
    t_v = ((- (n_taps - 1) / 2 : 1 : (n_taps - 1) / 2) .* Ts + t0);
    tn_v = t_v * 2/T;
    
    % Filter taps
    h_v = sinc(tn_v) ...
        .* (cos(pi .* rolloff .* tn_v )) ...
        ./ (1 - (2 * rolloff .* tn_v) .^ 2);
    
    h_v = h_v ./ sum(h_v);
end

function plot_cwdm4_D_and_lanes()
    lambda_nm_v = 1260 : 1342;

    x_lane_0 = [1261, 1281];
    x_lane_1 = [1281, 1301];
    x_lane_2 = [1301, 1321];
    x_lane_3 = [1321, 1341];

    y_up_lanes = [20, 20];
    basevalue = -10;
    y_max = + min(y_up_lanes + basevalue);
    y_min = - min(y_up_lanes + basevalue);

    D_up = get_cd_coeffitient_itu(lambda_nm_v, 0);
    D_down = get_cd_coeffitient_itu(lambda_nm_v, 1);
    D_worst = get_cd_coeffitient_itu(lambda_nm_v, 2);
    D_better = get_cd_coeffitient_itu(lambda_nm_v, 3);

    figure;
    hold all; box on; grid on;

    alpha = 0.25;

    h = area(x_lane_0, y_up_lanes, basevalue);
    h.EdgeColor = 'w';
    h.FaceColor = 'b';
    h.FaceAlpha = alpha;

    h = area(x_lane_1, y_up_lanes, basevalue);
    h.EdgeColor = 'w';
    h.FaceColor = 'g';
    h.FaceAlpha = alpha;

    h = area(x_lane_2, y_up_lanes, basevalue);
    h.EdgeColor = 'w';
    h.FaceColor = 'y';
    h.FaceAlpha = alpha;

    h = area(x_lane_3, y_up_lanes, basevalue);
    h.EdgeColor = 'w';
    h.FaceColor = 'r';
    h.FaceAlpha = alpha;

    plot(lambda_nm_v, D_up, '-r', "Linewidth", 5);
    plot(lambda_nm_v, D_down, '-b', "Linewidth", 5);
    plot(lambda_nm_v, D_worst, '--m', "Linewidth", 5);
    plot(lambda_nm_v, D_better, '--g', "Linewidth", 5);

    xlim([min(lambda_nm_v), max(lambda_nm_v)]);
    ylim([y_min, y_max]);
    tit = "ITU-T G.652 type fiber CD coefficient";
    title(tit,'Interpreter','latex', 'FontSize', 14);
    xlabel('$\lambda$ [$nm$]','Interpreter','latex', 'FontSize', 14);
    ylabel('D [ps/(nm $\cdot$ km)]','Interpreter','latex', 'FontSize', 14);
    set(gcf, 'Position', [50, 50, 550, 500], 'Color', 'w');
    leg = ["Lane 0", "Lane 1", "Lane 2", "Lane 3", "D up", "D down", "$|D|$ max", "$|D|$ min"];
    l = legend(leg, 'Interpreter','latex', 'FontSize', 14, 'Location', 'nw', "NumColumns", 2);
    title(l, 'IMDD CWDM4');
end
