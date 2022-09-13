clc

%% Network defintion
layers = get_lenet();
layers{1}.batch_size = 1;

% load the trained weights
load lenet.mat

path = '../images/'; 
% Read images from Images folder
all_img = dir(path);
for j=1:length(all_img)
    img_name = all_img(j).name;
    img_file = fullfile(path, img_name);
    try
        %Read Image
        img = imread(img_file);
        im = img(:,:,1);
        im=imresize(im, 3);

        %Show image
        figure(1)
        imshow(im);
        title('Input image')

        %Color to gray
        if size(im,3)==3 % RGB image
            im=rgb2gray(im);
        end

        %Binarize image
        threshold = graythresh(im);
        im =~imbinarize(im,threshold);

        %Remove all object containing fewer than 30 pixels
        im = bwareaopen(im,30);
        pause(3)

        %Display binary image
        figure(2)
        imshow(~im);
        title('Input image after bounding box')

        %Label connected components
        [L, Ne]=bwlabel(im);

        %Measure properties of image regions
        propied=regionprops(L,'BoundingBox');
        hold off

        %Plot Bounding Box
        for n=1:size(propied,1)
            rectangle('Position',propied(n).BoundingBox,'EdgeColor','r','LineWidth',1);
        end
        
        pause (3)
        figure;
        test_pred = zeros(784, size(propied,1));
        %%Plot Bounded 
        for n=1:size(propied,1)
            coord = propied(n).BoundingBox;
            subImage = imcrop(L, [coord(1), coord(2), coord(3), coord(4)]);
            subImage = imresize(subImage, [28, 28]);
            test_pred(:,n) =  im2col(subImage, [28 28]);
            subplot(1,size(propied,1),n), imshow(subImage);
            sgtitle('Bounded resized (28x28) digits')
        end
        
        %hold off
        pause (3)


        %% Testing the network
        num_digits = 10;
        prediction = zeros(num_digits, size(propied,1));
        for i=1:size(propied,1)
            im = test_pred(:,i);
            im = im(:);
            im = im./max(im);
            [output, P] = convnet_forward(params, layers, im);
            prediction(:, i) = P;
        end
        
        [val, ind] = max(prediction);
        fprintf('\n\n'); disp(thisname); fprintf('output\n')
        disp(ind)
   catch
   end
end



