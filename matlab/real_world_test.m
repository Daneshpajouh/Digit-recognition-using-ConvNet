clc
%% Network defintion
layers = get_lenet();
layers{1}.batch_size = 1;
% load the trained weights
load lenet.mat

%% Testing the network
num_digits = 10;
test_labels = [1, 3, 5, 8, 9, 6, 2];
test_pred = zeros(num_digits,size(test_labels, 2));
for i=1:size(test_labels, 2)
    imfile = "../test/sample_" + i + ".png";
    im = double(rgb2gray(imread(imfile))); % double and convert to grayscale
    im = imresize(im,[28,28]);  % change to 28 by 28 dimension
    %im = im(:); % unroll matrix to vector
    im = im2col(im, [28 28]);
    im = im./max(im);
    [output, P] = convnet_forward(params, layers, im);
    test_pred(:, i) = P;
end
disp(test_pred)
[val, ind] = max(test_pred);
cm = confusionchart(test_labels, ind-1);