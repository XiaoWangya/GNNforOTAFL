function [loss] = get_loss(W, X, y, param)
%% ========================================
%   Multinomial logistic regression
%       f: Regularized cross-entropy loss
%   --------------------------------------------
%   W: [w1, w2, ..., wc] weights for c classes
%   X: [x1, x2, ..., xn] n data for training
%   y: [y1; y2; ...; yn] n labels
%% ========================================
bs = length(y);
uni_label = unique(y);
loss = 0;
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
    
    tmp = find(y(i) == uni_label);
    den = sum(((expo([1:(tmp - 1), (tmp + 1):end])) / expo(tmp)) .^ (10^max_mag));
    loss = loss - log(1 / (1 + den));
end
loss = loss / bs;
reg = 0;
for c = 1:param.dim_label
    reg = reg + (norm(W(:, c))^2);
end

loss = loss + param.lamb * reg;

end