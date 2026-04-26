beta_candidates = linspace(-2,2,50) + 1i*linspace(-2,2,50).';
beta_candidates = beta_candidates(:);   % complex grid

best_beta = 0;
min_entropy = inf;

for b = 1:length(beta_candidates)
    
    beta = beta_candidates(b);
    expected_entropy = 0;
    
    % Consider possible photon counts (truncate)
    for m = 0:5
        
        % Compute probability of m
        P_m = 0;
        likelihoods = zeros(1,M);
        
        for j = 1:M
            lambda = abs(alpha_k(j) - beta)^2;
            likelihoods(j) = (lambda^m / factorial(m)) * exp(-lambda);
            P_m = P_m + P(j) * likelihoods(j);
        end
        
        if P_m < 1e-6
            continue;
        end
        
        % Posterior after observing m
        P_post = (P .* likelihoods) / P_m;
        
        % Entropy
        H = -sum(P_post .* log2(P_post + 1e-12));
        
        expected_entropy = expected_entropy + P_m * H;
    end
    
    % Keep best beta
    if expected_entropy < min_entropy
        min_entropy = expected_entropy;
        best_beta = beta;
    end
end

beta = best_beta;