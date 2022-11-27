function [acc] = get_acc(W, X, y, param)
%% ========================================
%   Multinomial logistic regression
%       f: Regularized cross-entropy loss
%   --------------------------------------------
%   W: [w1, w2, ..., wc] weights for c classes
%   X: [x1, x2, ..., xn] n data for test
%   y: [y1; y2; ...; yn] n labels
%% ========================================
bs = length(y);
cnt = 0;
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
    expo = exp(expo / 10^max_mag);
    
    pred_prob = zeros(param.dim_label, 1);
    for c = 1:param.dim_label
        den = sum(((expo([1:(c - 1), (c + 1):end])) / expo(c)) .^ (10^max_mag));
        pred_prob(c) = 1 / (1 + den);
    end
    y_hat = find(pred_prob == max(pred_prob));
    y_hat = param.label_total(y_hat(randi(length(y_hat))));
    if y(i) == y_hat
        cnt = cnt + 1;
    end
end
acc = cnt / bs;

end