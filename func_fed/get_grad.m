function [dW] = get_grad(W, X, y, param)
%% ========================================
%   Multinomial logistic regression
%       f: Regularized cross-entropy loss
%   --------------------------------------------
%   W: [w1, w2, ..., wc] weights for c classes
%   X: [x1, x2, ..., xn] n data for training
%   y: [y1; y2; ...; yn] n labels
%% ========================================
bs = length(y);
dW = zeros(size(W));
for i = 1:bs
    expo = zeros(param.dim_label, 1);
    max_mag = 0;
    for c = 1:param.dim_label
        expo(c) = W(:, c)' * X(:, i) + param.bias;
        if abs(expo(c)) > 1e-6
            mag = floor(log10(abs(expo(c))));
            if mag > max_mag
                max_mag = mag;
            end
        end
    end
    expo = exp(expo / (10^max_mag));
    
    for c = 1:param.dim_label
        den = sum(((expo([1:(c - 1), (c + 1):end])) / expo(c)) .^ (10^max_mag));
        dW(:, c) = dW(:, c) - ((y(i) == param.label_total(c)) - 1 / (1 + den)) * X(:, i);
    end
end
dW = dW / bs + 2 * param.lamb * W;

end