function [param] = fed_initial(data_type, dist_type, param)
%%
if strcmp(data_type, 'mnist')
    param.lr = 0.1;
elseif strcmp(data_type, 'cifar10')
    param.lr = 0.015;
else
    error('Wrong data type.');
end

%% initial datasets
[X_train, y_train, X_test, y_test] = load_data(data_type);

param.dim_label = length(unique(y_test));
param.label_total = unique(y_test);
[param.dim_feature, bs_tot] = size(X_train);
bs = rand(param.num_dev, 1) * floor(bs_tot / param.num_dev);
param.weight = bs ./ bs_tot;

shards_per_dev = 10;
param.data.train_local = assign_data(X_train, y_train, dist_type, ...
    param.num_dev, shards_per_dev);
% if strcmp(dist_type, 'noniid')
%     shards_per_dev = 2;
% else
%     shards_per_dev = 1;
% end
% param.data.train_local = assign_data_new(X_train, y_train, ...
%     param.num_dev, shards_per_dev);

param.data.train.features = X_train;
param.data.train.labels = y_train;

param.data.test.features = X_test;
param.data.test.labels = y_test;

%%
param.W_ini = zeros(param.dim_feature, param.dim_label);
param.mini_bs = 0.3; % (0, 1)
param.local_ep = 1;
param.bias = 0;
param.lamb = 0;

end
