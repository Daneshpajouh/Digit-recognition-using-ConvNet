clc

layers = get_lenet();
load lenet.mat
% load data
% Change the following value to true to load the entire dataset.
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);
xtrain = [xtrain, xvalidate];
ytrain = [ytrain, yvalidate];
m_train = size(xtrain, 2);
batch_size = 64;
 
 
layers{1}.batch_size = 1;
img = xtest(:, 1);
img = reshape(img, 28, 28);
img = imresize(img, 10);
imshow(img')
pause(2)
figure;




%[cp, ~, output] = conv_net_output(params, layers, xtest(:, 1), ytest(:, 1));
output = convnet_forward(params, layers, xtest(:, 1));
output_1 = reshape(output{1}.data, 28, 28);
% Fill in your code here to plot the features.


%Visualization of feature maps of the second layer - CONV
output_2 = output{2}.data;
j = 1;

for i = 1:576:11520
    imgg = output_2(i:i+575);
    imgg = reshape(imgg, 24, 24);
    imgg = imresize(imgg, 5);
    imgg = imrotate(imgg, 270);
    imgg = flip(imgg, 2);
    subplot(4,5,j),imshow(imgg);
    sgtitle('Feature maps of the second layer - CONV')
    j = j+1;
end

pause(2)
figure;

%Visualization of feature maps of the third layer - ReLU
output_3 = output{3}.data;
j = 1;

for i = 1:576:11520
    imgg = output_3(i:i+575);
    imgg = reshape(imgg, 24, 24);
    imgg = imresize(imgg, 5);
    imgg = imrotate(imgg, 270);
    imgg = flip(imgg, 2);
    subplot(4,5,j),imshow(imgg);
    sgtitle('Feature maps of the third layer - ReLU')
    j = j+1;
end