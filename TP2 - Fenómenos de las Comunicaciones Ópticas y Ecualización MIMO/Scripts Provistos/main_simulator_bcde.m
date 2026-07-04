clear all
close all

%%
% Parametros TX
Lsim=500e3; % Symbols
M=4; % QPSK=4, QAM16=16
BR = 32e9; % Symbol rate

% Pametros generales
NRX=2; % Receiver oversampling ratio
NOS=2; % Channel oversampling ratio

% Parametros del canal
Lfiber=100; %km
bcde_Lfiber=100; % Esta surge de un algoritmo de estimacion de long de fibra
DGD_ps = 50;
SOPMD_ps2 = 60*DGD_ps;
PCD_ps2 = 30*DGD_ps;

% Ruido
sigma=0;

% Parametros del receptor
Ntaps=21;
step=4e-3;
target_agc=0.3;

% Estructura
% TX --> Pulse Shaping --> Canal PMD --> Ruido --> Antialias + decim
% (opcional) --> AGC --> FFE MIMO --> decim --> BER

%% TRANSMISOR
pulse_shaping_type=0;
% Pulse shaping con RRC-RC
if pulse_shaping_type==0
    type='sqrt';
else
    type='normal';
end
htx = rcosine(1, NOS, type ,0.1, 15); 

% TX H
dec_labels_h = randi([0 M-1], Lsim, 1);
tx_symbs_h = qammod(dec_labels_h,M);
xup_h = upsample(tx_symbs_h,NOS);
sh = filter(htx,1,xup_h);

% TX V
dec_labels_v = randi([0 M-1], Lsim, 1);
tx_symbs_v = qammod(dec_labels_v,M);
xup_v = upsample(tx_symbs_v,NOS);
sv = filter(htx,1,xup_v);

clear xup_h dec_labels_h dec_labels_v xup_v


%% Canal
config_channel=struct();
channel_input=struct();

% Load inputs
channel_input.h_signal_v = sh;
channel_input.v_signal_v = sv;

% Load config
config_channel.OVR               = NOS         ; % Oversampling factor
config_channel.BR                = BR     ; % Symbol Rate [Bd]

config_channel.CD_D_ps_nm_km     = 20;
config_channel.lambda            = 1550e-9;
config_channel.link_len_m        = Lfiber*1000; % Fiber length [m]

config_channel.DGD_ps            = DGD_ps; % Differential Group Delay [ps]
config_channel.SOPMD_ps2         = SOPMD_ps2         ; % Polarization Mode Dispersion [ps^2]
config_channel.SOPMD_CD_ps2      = PCD_ps2         ; % Chromatic Dispersion due PMD [ps^2]

config_channel.fSOP_tx           = 0e3       ; % Tx Rot of the States of Pol [Hz]
config_channel.fSOP_rx           = 00e3       ; % Rx Rot of the States of Pol [Hz]

config_channel.osnr_db           = 22        ; % Channel SNR [dB] --> PENDING
config_channel.en_noise          = 0         ; % 0:OFF | 1:ON

[o_data_s, ~] = m_channel(channel_input, config_channel);
h11=o_data_s.h11;
h12=o_data_s.h12;
h21=o_data_s.h21;
h22=o_data_s.h22;
ych_h = o_data_s.h_signal_v;
ych_v = o_data_s.v_signal_v;

% 
% n_taps = length(h11);
% n_v = 1 : n_taps;
% nfft=2^12;
% fs=NOS*BR;
% f_v = 0:fs/nfft:fs-fs/nfft;
% 
% 
% figure('Color', 'w');
% subplot 221
% plot(n_v, real(h11), '-or', n_v, imag(h11), '-ob');
% grid on; hold on;
% xlim([0, n_taps])
% title('$h_{11}(t)$','Interpreter','latex', 'FontSize', 14);
% xlabel('Samples','Interpreter','latex', 'FontSize', 14);
% ylabel('Amplitude','Interpreter','latex', 'FontSize', 14);
% legend({'Real', 'Imag'}, 'Interpreter','latex', ...
%                          'FontSize', 14, 'Location', 'se');    
% 
% subplot 222
% plot(n_v, real(h12), '-or', n_v, imag(h12), '-ob');
% grid on; hold on;
% xlim([0, n_taps])
% title('$h_{12}(t)$','Interpreter','latex', 'FontSize', 14);
% xlabel('Samples','Interpreter','latex', 'FontSize', 14);
% ylabel('Amplitude','Interpreter','latex', 'FontSize', 14);
% legend({'Real', 'Imag'}, 'Interpreter','latex', ...
%                          'FontSize', 14, 'Location', 'se');
% 
% subplot 223
% plot(n_v, real(h21), '-or', n_v, imag(h21), '-ob');
% grid on; hold on;
% xlim([0, n_taps])
% title('$h_{21}(t)$','Interpreter','latex', 'FontSize', 14);
% xlabel('Samples','Interpreter','latex', 'FontSize', 14);
% ylabel('Amplitude','Interpreter','latex', 'FontSize', 14);
% legend({'Real', 'Imag'}, 'Interpreter','latex', ...
%                          'FontSize', 14, 'Location', 'se');
%         
% subplot 224
% plot(n_v, real(h22), '-or', n_v, imag(h22), '-ob');
% grid on; hold on;
% xlim([0, n_taps])
% title('$h_{22}(t)$','Interpreter','latex', 'FontSize', 14);
% xlabel('Samples','Interpreter','latex', 'FontSize', 14);
% ylabel('Amplitude','Interpreter','latex', 'FontSize', 14);
% legend({'Real', 'Imag'}, 'Interpreter','latex', ...
%                          'FontSize', 14, 'Location', 'se');

%% ANTIALIAS (OPCIONAL) solo si hacemos cambiod de tasa
y_aaf_h = ych_h;
y_aaf_v = ych_v;

%% AGC
metric_h = std(y_aaf_h);
metric_v = std(y_aaf_v);
y_agc_h = y_aaf_h/metric_h*target_agc;
y_agc_v = y_aaf_v/metric_v*target_agc;

%% ECUALIZACION BCDE
% Primero calculo la rta en fcia del BCDE dado bcde_Lfiber
if bcde_Lfiber>0
    config_bcde.lambda = 1550; % 1550ns
    config_bcde.Dfiber = 0.02; % 20ps/nm/km, preferentemente deberia coincidir con el canal
    config_bcde.fftsize = 8*1024; % Arranco con una FFT grande x las dudas
    config_bcde.BR = BR;
    config_bcde.OSR = NRX;
    bcde_freq_resp = get_bcde_response(config_bcde, bcde_Lfiber); % Rta en frecuencia con FFT muy grande
    
    % Recortamos la respuesta para reducir complejidad
    bcde_imp_resp_full = fftshift(ifft(bcde_freq_resp));
    accum_energy = cumsum(abs(bcde_imp_resp_full).^2);
    normalized_energy = accum_energy/accum_energy(end);
    mm = find( (normalized_energy>(1-99.99/100)) & (normalized_energy<(99.99/100)));
    bcde_imp_resp_trim = bcde_imp_resp_full(mm);
    required_ntaps_bcde = length(bcde_imp_resp_trim);
    
    y_bcde_h = filter(bcde_imp_resp_trim,1,y_agc_h);
    y_bcde_v = filter(bcde_imp_resp_trim,1,y_agc_v);
    
else
    y_bcde_h = y_agc_h;
    y_bcde_v = y_agc_v;
end
    

%% Ecualizador MIMO
Ls=length(y_agc_h);
eq_buffer_h = zeros(Ntaps,1);
eq_buffer_v = zeros(Ntaps,1);
heq_00=zeros(Ntaps,1);
heq_01=zeros(Ntaps,1);
heq_10=zeros(Ntaps,1);
heq_11=zeros(Ntaps,1);
heq_00 ((Ntaps+1)/2)=1;
heq_11 ((Ntaps+1)/2)=1;

yh_log = zeros(Ls,1);
yv_log = zeros(Ls,1);
error_h_log = zeros(ceil(Ls/NRX),1);
error_v_log = zeros(ceil(Ls/NRX),1);
zh_log = zeros(ceil(Ls/NRX),1);
zv_log = zeros(ceil(Ls/NRX),1);
dec_h_log = zeros(ceil(Ls/NRX),1);
dec_v_log = zeros(ceil(Ls/NRX),1);

subsf=1e3/2;
h00_log = zeros(ceil(Ls/NRX/subsf), Ntaps);
h01_log = zeros(ceil(Ls/NRX/subsf), Ntaps);
h10_log = zeros(ceil(Ls/NRX/subsf), Ntaps);
h11_log = zeros(ceil(Ls/NRX/subsf), Ntaps);

for n=1:Ls
    
    % Meto nueva muestra en el buffer
    eq_buffer_h(2:end)=eq_buffer_h(1:end-1);
    eq_buffer_h(1) = y_bcde_h(n);
    eq_buffer_v(2:end)=eq_buffer_v(1:end-1);
    eq_buffer_v(1) = y_bcde_v(n);
    
    % Producto MIMO
    yh = sum(eq_buffer_h.*heq_00) + sum(eq_buffer_v.*heq_01);
    yv = sum(eq_buffer_h.*heq_10) + sum(eq_buffer_v.*heq_11);
    yh_log(n) = yh;
    yv_log(n) = yv;
    
    if mod(n,NRX)==0
        
        n2=floor(n/NRX)+1;
        n3=floor(n2/subsf)+1;
        
        zh_log(n2)=yh;
        zv_log(n2)=yv;
        
        dec_h = slicer_qam(yh,M);
        dec_v = slicer_qam(yv,M);
        dec_h_log(n2) = dec_h;
        dec_v_log(n2) = dec_v;
        
        error_h = yh-dec_h;
        error_v = yv-dec_v;
        error_h_log(n2)=error_h;
        error_v_log(n2)=error_v;
        
        heq_00 = heq_00*(1-step*1e-2) - step*conj(eq_buffer_h)*error_h;
        heq_01 = heq_01*(1-step*1e-2) - step*conj(eq_buffer_v)*error_h;
        heq_10 = heq_10*(1-step*1e-2) - step*conj(eq_buffer_h)*error_v;
        heq_11 = heq_11*(1-step*1e-2) - step*conj(eq_buffer_v)*error_v;
        
        if mod(n2,subsf)==0
            h00_log(n3,:)=heq_00;
            h01_log(n3,:)=heq_01;
            h10_log(n3,:)=heq_10;
            h11_log(n3,:)=heq_11;
        end
    end
end

% Plot constellations
valid=fix(0.9*Lsim):Lsim-100;
figure
subplot 121
plot(zh_log(valid),'.')
grid on; axis equal; xlim([-2,2]); ylim([-2,2])
xlabel("Real Part"); ylabel("Imag Part"); title("Constellation H")
subplot 122
plot(zv_log(valid),'.')
grid on; axis equal; xlim([-2,2]); ylim([-2,2])
xlabel("Real Part"); ylabel("Imag Part"); title("Constellation V")

% Evolution of symbols
figure
subplot 221
plot(real(zh_log),'.');
xlabel('Samples'); ylabel('Amplitude'); title('HI'); grid on
subplot 222
plot(imag(zh_log),'.');
xlabel('Samples'); ylabel('Amplitude'); title('HQ'); grid on
subplot 223
plot(real(zv_log),'.');
xlabel('Samples'); ylabel('Amplitude'); title('VI'); grid on
subplot 224
plot(imag(zv_log),'.');
xlabel('Samples'); ylabel('Amplitude'); title('VQ'); grid on

% Rta final FFE
figure
subplot 221
plot(real(heq_00),'-ob'); hold all
plot(imag(heq_00),'-or');
xlabel('Samples'); ylabel('Imp. Resp.'); title('H00'); grid on; ylim([-3,3])
subplot 222
plot(real(heq_01),'-ob'); hold all
plot(imag(heq_01),'-or');
xlabel('Samples'); ylabel('Imp. Resp.'); title('H01'); grid on; ylim([-3,3])
subplot 223
plot(real(heq_10),'-ob'); hold all
plot(imag(heq_10),'-or');
xlabel('Samples'); ylabel('Imp. Resp.'); title('H10'); grid on; ylim([-3,3])
subplot 224
plot(real(heq_11),'-ob'); hold all
plot(imag(heq_11),'-or');
xlabel('Samples'); ylabel('Imp. Resp.'); title('H11'); grid on; ylim([-3,3])


Ls=length(h00_log);
nline = subsf*(0:Ls-1)';
figure
subplot 221
plot(nline, abs(h00_log))
xlabel('Symbols'); ylabel('Abs. Value'); title('HH'); grid on
subplot 222
plot(nline, abs(h01_log))
xlabel('Symbols'); ylabel('Abs. Value'); title('HV'); grid on
subplot 223
plot(nline, abs(h10_log))
xlabel('Symbols'); ylabel('Abs. Value'); title('VH'); grid on
subplot 224
plot(nline, abs(h11_log))
xlabel('Symbols'); ylabel('Abs. Value'); title('VV'); grid on

figure
nfft=1024;
fs=NRX*BR/1e9;
dF=fs/nfft;
fv=0:dF:fs-dF;
subplot 221
plot(fv, 20*log10(abs(fft(heq_00, nfft))))
grid on; xlabel('Freq [GHz]'); ylabel('Mag. Freq. Resp.'); title('HH')
subplot 222
plot(fv, 20*log10(abs(fft(heq_01, nfft))))
grid on; xlabel('Freq [GHz]'); ylabel('Mag. Freq. Resp.'); title('HV')
subplot 223
plot(fv, 20*log10(abs(fft(heq_10, nfft))))
grid on; xlabel('Freq [GHz]'); ylabel('Mag. Freq. Resp.'); title('VH')
subplot 224
plot(fv, 20*log10(abs(fft(heq_11, nfft))))
grid on; xlabel('Freq [GHz]'); ylabel('Mag. Freq. Resp.'); title('VV')