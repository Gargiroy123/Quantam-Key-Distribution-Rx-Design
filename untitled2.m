clc; clear; close all;

M = 4;
alpha = 2;              % amplitude
sigma = 0.5;            % noise variance (controls blob size)

k = 0:M-1;
theta = pi*(2*k+1)/M;

figure; hold on; grid on;

% Create grid for contour
[x, y] = meshgrid(linspace(-3,3,200), linspace(-3,3,200));

for idx = 1:M
    
    % Center of coherent state
    alpha_k = alpha * exp(1i * theta(idx));
    x0 = real(alpha_k);
    y0 = imag(alpha_k);
    
    % Gaussian PDF (2D)
    Z = exp(-((x - x0).^2 + (y - y0).^2)/(2*sigma^2));
    
    % Contour plot (cloud boundary)
    contour(x, y, Z);
    
    % Fill center point
    plot(x0, y0, 'ko', 'MarkerFaceColor', 'k');
    
    % Draw arrow from origin
%     quiver(0, 0, x0, y0, 0, 'r', 'LineWidth', 1.5, 'MaxHeadSize', 0.5);
end

% Draw origin
plot(0,0,'ro','MarkerFaceColor','r');

xlabel('In-phase (X)');
ylabel('Quadrature (P)');
title('QPSK Coherent States with Gaussian Uncertainty');
axis equal;
xlim([-3 3]); ylim([-3 3]);