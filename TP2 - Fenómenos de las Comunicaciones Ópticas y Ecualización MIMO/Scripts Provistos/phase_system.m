clear all
%close all

fs = 16e9; %GHz (fs = BR para el FCR)
Ts = 1/fs;
z = tf('z', Ts);

%%
% System modelling
Kp = 0.002;
Ki = Kp/1000;

L = Kp + Ki*z/(z-1);
NCO = z/(z-1);
G = L*NCO;
Lat = 100;
H = z^-Lat;
F = feedback(G, H);

%% Step response
figure
step(F)

%% Ramp response
figure
step(F*z/(z-1))
hold all
step(z/(z-1))

% %%
figure
pzmap(F)

%%
figure
bode(F)
hold all
