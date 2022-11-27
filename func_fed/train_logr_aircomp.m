function [acc_set, loss_set] = train_logr_aircomp(param, channel, p, eta,sigma)
%%
acc_set = nan(1, param.comm + 1);
loss_set = nan(1, param.comm + 1);

W_global = param.W_ini;
W_local = cell(param.num_dev, 1);
for k = 1:param.num_dev
    W_local{k} = param.W_ini;
end

data = param.data;

acc_set(1) = get_acc(W_global, data.test.features, data.test.labels, param);
loss_set(1) = get_loss(W_global, data.train.features, data.train.labels, param);

%%
for t = 1:param.comm
    %% device selection
    dev_sel = min(param.num_dev, param.num_dev);
    dev_ind = randperm(param.num_dev, dev_sel);
    
    %%
    if dev_sel > 0
        %% model dissemination
        for k = 1:dev_sel
            ind = dev_ind(k);
            W_local{ind} = W_global;
        end
        
        %% local training
        for k = 1:dev_sel
            ind = dev_ind(k);
            W_local{ind} = train(W_local{ind}, data.train_local.features{ind}, ...
                data.train_local.labels{ind}, param);
        end
        
        %% uplink aggregation
        model = 0;
        weight_sum = 0;
        for k = 1:dev_sel
            ind = dev_ind(k);
            model = model + param.weight(ind) * (W_local{ind}*abs(channel(t,k))*sqrt(p(t,k))+randn()*sigma);
            weight_sum = weight_sum + param.weight(ind);
        end
        model = model / (weight_sum*sqrt(eta(t)));
        
        if strcmp(param.agg_type, 'grad')
            W_global = W_global - param.lr * model;
        elseif strcmp(param.agg_type, 'diff')
            W_global = W_global + model;
        else
            W_global = model;
        end
        
        %% model evaluation
        acc_set(t + 1) = get_acc(W_global, data.test.features, ...
            data.test.labels, param);
        loss_set(t + 1) = get_loss(W_global, data.train.features, ...
            data.train.labels, param);
        if param.verb
            if mod(t, param.rep)==0
                fprintf('Comm: %d - Test accuracy: %.4f, Training loss: %.3e\n', t, acc_set(t + 1), loss_set(t + 1));
            end
        end
        
    end
    
end
end

%% model training
function [model] = train(W, X, y, param)
bs = length(y);
if strcmp(param.agg_type, 'grad')
    ind = randperm(bs, floor(bs * param.mini_bs));
    model = get_grad(W, X(:, ind), y(ind), param);
else
    W0 = W;
    for ep = 1:param.local_ep
        ind = randperm(bs, floor(bs * param.mini_bs));
        dW = get_grad(W, X(:, ind), y(ind), param);
        W = W - param.lr * dW;
    end
    if strcmp(param.agg_type, 'diff')
        model = W - W0;
    else
        model = W;
    end
end
end
