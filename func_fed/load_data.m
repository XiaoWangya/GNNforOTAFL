function [X_train, y_train, X_test, y_test] = load_data(type)
if strcmp(type, 'mnist')
    X_train = loadImages('./data_mnist/train-images.idx3-ubyte');
    y_train = loadLabels('./data_mnist/train-labels.idx1-ubyte') + 1;
    
    X_test = loadImages('./data_mnist/t10k-images.idx3-ubyte');
    y_test = loadLabels('./data_mnist/t10k-labels.idx1-ubyte') + 1;
elseif strcmp(type, 'cifar10')
    data1 = load('./data_cifar10/data_batch_1.mat');
    data2 = load('./data_cifar10/data_batch_2.mat');
    data3 = load('./data_cifar10/data_batch_3.mat');
    data4 = load('./data_cifar10/data_batch_4.mat');
    data5 = load('./data_cifar10/data_batch_5.mat');
    X_train = double([data1.data; data2.data; data3.data; data4.data; data5.data])' / 255;
    y_train = double([data1.labels; data2.labels; data3.labels; data4.labels; data5.labels]) + 1;
    
    data_test = load('./data_cifar10/test_batch.mat');
    X_test = double(data_test.data)' / 255;
    y_test = double(data_test.labels) + 1;
else
    error('Wrong dataset type.');
end
end
%% load images
function [images] = loadImages(filename)
fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Cannot open the file.', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in.', filename, '']);

num_images = fread(fp, 1, 'int32', 0, 'ieee-be');
num_rows = fread(fp, 1, 'int32', 0, 'ieee-be');
num_cols = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'unsigned char');
images = reshape(images, num_cols, num_rows, num_images);
images = permute(images, [2, 1, 3]);

fclose(fp);

% two dimensional matrix, each col represents an image (matrix with (pixel * example))
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% normalization
images = double(images / 255);
end
%% load labels
function [labels] = loadLabels(filename)
fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Cannot open the file.', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in.', filename, '']);

numLabels = fread(fp, 1, 'int32',0,'ieee-be');

labels = fread(fp, inf, 'unsigned char');
assert(size(labels, 1) == numLabels, 'Mismatch in label count.');

fclose(fp);
end