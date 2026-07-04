clear all
%close all

%%
% Parametros TX
Lsim=250e3; % Symbols
M=16; % QPSK=4, QAM16=16
BR = 128e9; % Symbol rate

% Pametros generales
NRX=2; % Receiver oversampling ratio
NOS=2; % Channel oversampling ratio

% Parametros del canal
Lfiber=1000; %km
bcde_Lfiber=1000; % Esta surge de un algoritmo de estimacion de long de fibra
DGD_ps = 0;
SOPMD_ps2 = 60*DGD_ps;
PCD_ps2 = 30*DGD_ps;

% Parametros del receptor
Ntaps=63;
step=1e-3;
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
RCMA=sqrt(mean(abs(tx_symbs_h).^4)/mean(abs(tx_symbs_h).^2));

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
config_channel.fSOP_rx           = 0e3       ; % Rx Rot of the States of Pol [Hz]

config_channel.osnr_db           = 24        ; % Channel SNR [dB] --> PENDING
config_channel.en_noise          = 1         ; % 0:OFF | 1:ON

[o_data_s, ~] = m_channel(channel_input, config_channel);
h11=o_data_s.h11;
h12=o_data_s.h12;
h21=o_data_s.h21;
h22=o_data_s.h22;
ych_h = o_data_s.h_signal_v;
ych_v = o_data_s.v_signal_v;

% Agregar lo offset y ruido de fase
lo_offset=0e6;
LW=0e3;
Ls=length(ych_h); nline=(0:Ls-1)'; tline=nline/(NOS*BR);
laser_lo_offset=exp(1j*tline*2*pi*lo_offset);
wnoise=sqrt(2*pi*LW/(NOS*BR))*randn(Ls,1);
laser_pnoise=exp(1j*cumsum(wnoise));
ych_h=ych_h.*laser_lo_offset.*laser_pnoise;
ych_v=ych_v.*laser_lo_offset.*laser_pnoise;

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

%% Ecualizador BCDE
% Primero calculo la rta en fcia del BCDE dado bcde_Lfiber
if bcde_Lfiber>0
    config_bcde.lambda = 1550; % 1550ns
    config_bcde.Dfiber = 0.02; % 20ps/nm/km, preferentemente deberia coincidir con el canal
    config_bcde.fftsize = 128*1024; % Arranco con una FFT grande x las dudas
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
    
    % Usando convolucion
%     tic 
%     y_bcde_h = filter(bcde_imp_resp_trim,1,y_agc_h);
%     y_bcde_v = filter(bcde_imp_resp_trim,1,y_agc_v);
%     toc
    
    % Usar filtrado en el dominio de la frecuencia es lo mejor
    % Determino un size de FFT
    tentative_nfft = 2*required_ntaps_bcde;
    nfft = 2^(ceil(log2(tentative_nfft)));
    overlap = nfft/2; % Overlap de 50%
    rta_impulso = [bcde_imp_resp_trim, zeros(1,nfft-length(bcde_imp_resp_trim))];
    % Alineo rta impulso a la izquierda --> Descarto muestras a la izq en
    % el O&S
    rta_freq = fft(rta_impulso);
    block_size = nfft - overlap;
    Ls=floor(length(y_agc_h)/block_size)*block_size;
    nblocks = Ls/block_size;
    
    y_bcde_h=zeros(Ls,1);
    y_bcde_v=zeros(Ls,1);
    
    tic
    for nb=2:nblocks
        slice_input = 1+(nb-1)*block_size-overlap:(nb)*block_size;
        fft_input_h = y_agc_h(slice_input);
        fft_input_v = y_agc_v(slice_input);
        
        fft_output_h = fft(fft_input_h);
        fft_output_v = fft(fft_input_v);
        
        filter_out_h = fft_output_h.*rta_freq.';
        filter_out_v = fft_output_v.*rta_freq.';
        
        ifft_output_h = ifft(filter_out_h);
        ifft_output_v = ifft(filter_out_v);
        y_bcde_h(1+(nb-1)*block_size:nb*block_size)=ifft_output_h(1+overlap:end);
        y_bcde_v(1+(nb-1)*block_size:nb*block_size)=ifft_output_v(1+overlap:end);
    end
    toc
    
else
    y_bcde_h = y_agc_h;
    y_bcde_v = y_agc_v;
end

%% Ecualizacion MIMO
Ls=length(y_bcde_h);
eq_buffer_h = zeros(Ntaps,1);
eq_buffer_v = zeros(Ntaps,1);
heq_00=zeros(Ntaps,1);
heq_01=zeros(Ntaps,1);
heq_10=zeros(Ntaps,1);
heq_11=zeros(Ntaps,1);
heq_00 ((Ntaps+1)/2)=8;
heq_11 ((Ntaps+1)/2)=8;

yh_log = zeros(Ls,1);
yv_log = zeros(Ls,1);
error_h_log = zeros(ceil(Ls/NRX),1);
error_v_log = zeros(ceil(Ls/NRX),1);
ffe_output_h_log = zeros(ceil(Ls/NRX),1);
ffe_output_v_log = zeros(ceil(Ls/NRX),1);
dec_h_log = zeros(ceil(Ls/NRX),1);
dec_v_log = zeros(ceil(Ls/NRX),1);
fcr_output_h_log= zeros(ceil(Ls/NRX),1);
fcr_output_v_log= zeros(ceil(Ls/NRX),1);
fcr_nco= zeros(ceil(Ls/NRX),2);
fcr_inte=zeros(ceil(Ls/NRX),2);
kp_fcr=25e-3; ki_fcr=kp_fcr/1000;
fcr_latency=0;

subsf=1e3/2;
h00_log = zeros(ceil(Ls/NRX/subsf), Ntaps);
h01_log = zeros(ceil(Ls/NRX/subsf), Ntaps);
h10_log = zeros(ceil(Ls/NRX/subsf), Ntaps);
h11_log = zeros(ceil(Ls/NRX/subsf), Ntaps);

% SM
enable_cma=1; % 0 es DD, 1 es CMA
enable_fcr=0; 
normal_operation=0; % declarar que esta todo ok

timer_cma=ceil(0.2*Lsim);
timer_fcr=ceil(0.3*Lsim);
timer_dd=ceil(0.1*Lsim);
normal_op_timer=timer_cma+timer_fcr+timer_dd;
force_cma=false;

for n=2*fcr_latency+3:Ls
    
    % Meto nueva muestra en el buffer
    eq_buffer_h(2:end)=eq_buffer_h(1:end-1);
    eq_buffer_h(1) = y_bcde_h(n);
    eq_buffer_v(2:end)=eq_buffer_v(1:end-1);
    eq_buffer_v(1) = y_bcde_v(n);
    
    % Producto MIMO
    yffe_h = sum(eq_buffer_h.*heq_00) + sum(eq_buffer_v.*heq_01);
    yffe_v = sum(eq_buffer_h.*heq_10) + sum(eq_buffer_v.*heq_11);
    yh_log(n) = yffe_h;
    yv_log(n) = yffe_v;
    
    if mod(n,NRX)==0
        n2=floor((n-1)/NRX)+1;
        n3=floor((n2-1)/subsf)+1;
        
        if n2<timer_cma
            enable_cma=1;
            enable_fcr=0; 
            normal_operation=0;
        elseif n2<(timer_fcr+timer_cma)
            enable_cma=1;
            enable_fcr=1; 
            normal_operation=0;
        elseif n2<(timer_fcr+timer_cma+timer_dd)
            enable_cma=0;
            enable_fcr=1; 
            normal_operation=0;
        else 
            enable_cma=0;
            enable_fcr=1; 
            normal_operation=1;
        end
        
        ffe_output_h_log(n2)=yffe_h;
        ffe_output_v_log(n2)=yffe_v;
        
        yfcr_h=yffe_h*exp(-1j*fcr_nco(n2-fcr_latency,1));
        yfcr_v=yffe_v*exp(-1j*fcr_nco(n2-fcr_latency,2));
        fcr_output_h_log(n2)=yfcr_h;
        fcr_output_v_log(n2)=yfcr_v;
        
        if ~enable_cma || enable_fcr
            dec_h = slicer_QAM(yfcr_h,M);
            dec_v = slicer_QAM(yfcr_v,M);
        end
                
        if enable_cma || force_cma
            error_cma_h=yffe_h*(abs(yffe_h)-RCMA);
            error_cma_v=yffe_v*(abs(yffe_v)-RCMA);
            error_h_log(n2)=error_cma_h;
            error_v_log(n2)=error_cma_v;
            error_h=error_cma_h;
            error_v=error_cma_v;
        else
            dec_h_log(n2) = dec_h;
            dec_v_log(n2) = dec_v;
            error_dd_bb_h = yfcr_h-dec_h; % Base band
            error_dd_bb_v = yfcr_v-dec_v;
            error_h_log(n2)=error_dd_bb_h;
            error_v_log(n2)=error_dd_bb_v;
            error_dd_pb_h=error_dd_bb_h*exp(1j*fcr_nco(n2-fcr_latency,1)); % pass band error
            error_dd_pb_v=error_dd_bb_v*exp(1j*fcr_nco(n2-fcr_latency,2));
            error_h=error_dd_pb_h;
            error_v=error_dd_pb_v;
        end
        
        if enable_fcr
            yfcr(1)=yfcr_h;
            yfcr(2)=yfcr_v;
            dec(1)=dec_h;
            dec(2)=dec_v;
            for pol=1:2
               phase_error = -angle(dec(pol))+angle(yfcr(pol));
               prop_error=kp_fcr*phase_error;
               fcr_inte(n2+1,pol)=fcr_inte(n2,pol)+ki_fcr*phase_error;
               fcr_nco(n2+1,pol)=fcr_nco(n2,pol)+fcr_inte(n2+1,pol)+prop_error;
            end
        end
        
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
%%
% Plot constellations
valid=fix(0.8*Lsim):fix(0.9*Lsim);
figure
subplot 121
plot(fcr_output_h_log(valid),'.')
grid on; axis equal; xlim([-4,4]); ylim([-4,4])
xlabel("Real Part"); ylabel("Imag Part"); title("Constellation H")
subplot 122
plot(fcr_output_v_log(valid),'.')
grid on; axis equal; xlim([-4,4]); ylim([-4,4])
xlabel("Real Part"); ylabel("Imag Part"); title("Constellation V")

% Evolution of symbols
figure
subplot 221
plot(real(fcr_output_h_log),'.');
xlabel('Samples'); ylabel('Amplitude'); title('HI'); grid on
subplot 222
plot(imag(fcr_output_h_log),'.');
xlabel('Samples'); ylabel('Amplitude'); title('HQ'); grid on
subplot 223
plot(real(fcr_output_v_log),'.');
xlabel('Samples'); ylabel('Amplitude'); title('VI'); grid on
subplot 224
plot(imag(fcr_output_v_log),'.');
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

%% Differential correlator
Ltrim=ceil(Lsim/2);
ffe_output_trim(:,1)=fcr_output_h_log(Ltrim:end-1e3);
ffe_output_trim(:,2)=fcr_output_v_log(Ltrim:end-1e3);
tx_trim(:,1)=tx_symbs_h(Ltrim:end-1e3);
tx_trim(:,2)=tx_symbs_v(Ltrim:end-1e3);

Lcorr=20000;
rx_diff(:,1)=ffe_output_trim(2:Lcorr,1).*conj(ffe_output_trim(1:Lcorr-1,1));
tx_diff(:,1)=tx_trim(2:Lcorr,1).*conj(tx_trim(1:Lcorr-1,1));
rx_diff(:,2)=ffe_output_trim(2:Lcorr,2).*conj(ffe_output_trim(1:Lcorr-1,2));
tx_diff(:,2)=tx_trim(2:Lcorr,2).*conj(tx_trim(1:Lcorr-1,2));

% Este correlador no puede arreglar reflex (cambios de signo por lane en
% TX)
corrs=cell(2,2);
swap=false;
figure
for ntx=1:2
    max_corr=0;
    best_nrx(ntx)=1;
    for nrx=1:2
        subplot(2,2,(ntx-1)*2+nrx)
        corrs{ntx,nrx}=abs(xcorr(tx_diff(:,ntx),rx_diff(:,nrx)))/Lcorr/var(tx_symbs_h(2:Lcorr));
        plot(corrs{ntx,nrx});
        grid on
        if max(corrs{ntx,nrx})>max_corr
            max_corr=max(corrs{ntx,nrx});
            best_nrx(ntx)=nrx;
        end
    end
end

if best_nrx==[1 2]
    swap=false;
    colision=false;
elseif best_nrx==[2 1]
    swap=true;
    colision=false;
else
    swap=false;
    colision=true;
end
assert(~colision)

delay_h = finddelay(tx_diff(:,1), rx_diff(:,best_nrx(1)));
delay_v = finddelay(tx_diff(:,2), rx_diff(:,best_nrx(2)));
rx_align(:,1) = ffe_output_trim(1+delay_h:end,1);
rx_align(:,2) = ffe_output_trim(1+delay_v:end,2);
tx_align(:,1) = tx_trim(1:length(rx_align(:,1)),best_nrx(1)); 
tx_align(:,2) = tx_trim(1:length(rx_align(:,2)),best_nrx(2)); 

% En este punto tengo las seniales alineadas
% Ejecuto el CSC dinamico
csc_block_size = 128;
Nblocks = floor(length(rx_align)/csc_block_size);
rx_fix_csc = zeros(Nblocks*csc_block_size,2);
for pol=1:2
    for nb=1:Nblocks
       csc_rx_in = rx_align(1+(nb-1)*csc_block_size : nb*csc_block_size,pol);
       csc_tx_in = tx_align(1+(nb-1)*csc_block_size : nb*csc_block_size,pol);
       best_p=0;
       best_mse=inf;
       for ptest=[0,pi/2,pi,1.5*pi]
           mse=mean(abs(csc_tx_in-csc_rx_in.*exp(1j*ptest)).^2);
           if mse < best_mse
               best_mse=mse;
               best_p = ptest;
           end
       end
       csc_rx_out = csc_rx_in.*exp(1j*best_p);
       rx_fix_csc(1+(nb-1)*csc_block_size : nb*csc_block_size,pol)=csc_rx_out;
    end
end
tx_fix_csc=tx_align(1:Nblocks*csc_block_size,:);


%%
for pol=1:2
    rx_decs = slicer_QAM(rx_fix_csc(:,pol),M);
    errors_symb = sum(rx_decs~=tx_fix_csc(:,pol));
    ser = errors_symb/length(rx_decs);
    ber_per_pol(pol) = 1/log2(M)*ser;
end
ber=mean(ber_per_pol)

