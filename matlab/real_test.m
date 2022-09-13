clear
clc
%% Network defintion
layers = get_lenet();

% load the trained weights
load lenet.mat

%% Testing the network
num_digits = 10;
test_labels = [1, 3, 5, 8, 9];
input_data = zeros(28*28,size(test_labels, 2));
test_pred = zeros(num_digits,size(test_labels, 2));
for i=1:size(test_labels, 2)
    imfile = "../test/sample_" + i + ".png";
    im = double(rgb2gray(imread(imfile))); % double and convert to grayscale
    im = imresize(im,[28,28]);  % change to 28 by 28 dimension
    im = im(:); % unroll matrix to vector
    im = im./max(im);
    input_data(:, i) = im;
end
for i=1:5:size(test_labels, 2)
    [output, P] = convnet_forward(params, layers, input_data(:, i:i+4));
    test_pred(:, i:i+4) = test_pred(:, i:i+4) + P;
end

[val, ind] = max(test_pred);
cm = confusionchart(ind-1, test_labels);