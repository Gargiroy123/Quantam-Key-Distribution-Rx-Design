clc; clear; close all;

%% 🔷 PARAMETERS
alpha = 1;          % amplitude
Nsamples = 10000;      % number of samples per state
sigma = 1;     % shot noise variance (SNU)
M = 4;
k = 0:M-1
theta = pi*(2*k+1)/M;
%% 🔷 QPSK STATES

alpha_k = alpha * exp(1i * theta);

figure; hold on; grid on;

%% 🔷 GENERATE CLOUDS
for j = 1:2
    
    center = alpha_k(j);
    
    % Gaussian noise (heterodyne measurement)
    noise = (randn(Nsamples,1) + 1i*randn(Nsamples,1)) * sqrt(sigma/2);
    
    % Samples (cloud)
    samples = center + noise;
    
    % Plot cloud
    scatter(real(samples), imag(samples), 10, 'filled');
    
    % Plot center
    plot(real(center), imag(center), 'ko', 'MarkerSize', 8, 'LineWidth', 2);
end

%% 🔷 DRAW AXES AND LABELS
plot(0,0,'ro','MarkerFaceColor','r');  % origin

xlabel('In-phase (X)');
ylabel('Quadrature (P)');
title('BPSK Coherent States as Quantum Gaussian Clouds');

axis equal;
xlim([-4 4]);
ylim([-4 4]);
legend('+\alpha cloud', '-\alpha cloud');