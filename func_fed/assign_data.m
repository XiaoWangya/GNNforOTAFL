function [data] = assign_data(X, y, dist_type, num_dev, shards_per_dev)
%%
data.features = cell(num_dev, 1);
data.labels = cell(num_dev, 1);
if strcmp(dist_type, 'noniid')
    [y, ind] = sort(y);
    X = X(:, ind);
    shards_total = num_dev * shards_per_dev;
    bs = floor(length(y) / shards_total);
    shards_set = cell(shards_total, 1);
    ind = 1;
    for i = 1:shards_total
        shards_set{i} = [X(:, ind:(ind + bs - 1)); y(ind:(ind + bs - 1))'];
        ind = ind + bs;
    end
    shards_ind = randperm(shards_total);
    for k = 1:num_dev
        tmp = [];
        for i = 1:shards_per_dev
            tmp = [tmp, shards_set{shards_ind(shards_per_dev * (k - 1) + i)}];
        end
        data.features{k} = tmp(1:(end - 1), :);
        data.labels{k} = tmp(end, :)';
    end
else
    bs = floor(length(y) / num_dev);
    ind = 1;
    for k = 1:num_dev
%         data.features{k} = X(:, ind:(ind + bs - floor(bs*0.8)));
        data.features{k} = X(:, ind:(ind + bs - 1));
        data.labels{k} = y(ind:(ind + bs - 1));
%         data.labels{k} = y(ind:(ind + bs - floor(bs*0.8)));
        ind = ind + bs;
    end
end

end