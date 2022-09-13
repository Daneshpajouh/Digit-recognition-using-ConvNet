# Digit recognition using ConvNet
### Spring 2021
*Course project (Permission granted from the instructor)

Convolutional Neural Network (CNN) for building a numeric character recognition system trained on the MNIST dataset (Written in Matlab).


# Overview

A typical convolutional neural network has four different types of layers.
### Fully Connected Layer / Inner Product Layer (IP)
The fully connected or the inner product layer is the simplest layer which makes up neural networks. Each neuron of the layer is connected to all the neurons of the previous layer (See Fig 1). Mathematically it is modelled by a matrix multiplication and the addition of a bias term. For a given input x the output of the fully connected layer is given by the following equation,

<p align="center">
f (x) = W x + b
</p>

W, b are the weights and biases of the layer. W is a two dimensional matrix of m × n size where n is the dimensionality of the previous layer and m is the number of neurons in this layer. b is a vector with size m × 1.

<p align="center">
  <img width="150" alt="image" src="https://user-images.githubusercontent.com/52765136/189790535-076c62f8-71e2-4c84-8283-566e18830d76.png">
</p>
<p align="center">
  <img width="170" alt="image" src="https://user-images.githubusercontent.com/52765136/189790664-d37af749-2ada-4008-8257-3789ae460513.png">
</p>


### Convolutional Layer
This is the fundamental building block of CNNs. Before we delve into what a convolution layer is, let’s take a quick look at convolution.

Convolution is performed using a k × k filter/kernel and a W × H image. The output of the convolution operation is a feature map. This feature map can bear different meanings according to the filters being used - for example, using a Gaussian filter will lead to a blurred version of the image. Using the Sobel filters in the x and y direction give us the corresponding edge maps as outputs.

**Terminology** : Each number in a filter will be referred to as a filter weight. For example, the 3x3 gaussian filter has the following 9 filter weights.

<p align="center">
  <img width="193" alt="image" src="https://user-images.githubusercontent.com/52765136/189791255-9b21c9b1-da3d-4d54-869a-e4b1a2429aba.png">
</p>

When we perform convolution, we decide the exact type of filter we want to use and accordingly decide the filter weights. CNNs try to learn these filter weights and biases from the data. We attempt to learn a set of filters for each convolutional layer.

In general there are two main motivations for using convolution layers instead of fully- connected (FC) layers (as used in neural networks).

**1. A reduction in parameters**
  
In FC layers, every neuron in a layer is connected to every neuron in the previous layer. This leads to a large number of parameters to be estimated - which leads to over-fitting. CNNs change that by sharing weights (the same filter is translated over the entire image).

**2.	It exploits spatial structure**

Images have an inherent 2D spatial structure, which is lost when we unroll the image into a vector and feed it to a plain neural network. Convolution by its very nature is a 2D operation which operates on pixels which are spatially close.

**Implementation details**: The general convolution operation can be represented by the following equation:

<p align="center">
f (X, W, b) = X ∗ W + b
</p>

where W is a filter of size k×k×C<sub>i</sub>, X is an input volume of size N<sub>i</sub> ×N<sub>i</sub> ×C<sub>i</sub> and b is 1×1 element. The meanings of the individual terms are shown below.

<p align="center">
<img width="568" alt="image" src="https://user-images.githubusercontent.com/52765136/189793413-31bd417d-7245-44e7-be36-134ed646ea74.png">
</p>

In the following example the subscript i refers to the input to the layer and the subscript o refers to the output of the layer.

* N<sub>i</sub> - width of the input image

* N<sub>i</sub> - height of the input image (image has a square shape)

* C<sub>i</sub> - number of channels in the input image

* k<sub>i</sub> - width of the filter

* s<sub>i</sub> - stride of the convolution

* p<sub>i</sub> - number of padding pixels for the input image

* num - number of convolution filters to be learnt

A grayscale image has 1 channel, which is the depth of the image volume. For an image with C<sub>i</sub> channels - we will learn num filters of size k<sub>i</sub> × k<sub>i</sub> × C<sub>i</sub>. The output of convolving with each filter is a feature map with height and width N<sub>o</sub>, where

<p align="center">
<img width="169" alt="image" src="https://user-images.githubusercontent.com/52765136/189805490-1fb42f80-7407-435f-924e-0fdeed9f6dec.png">
</p>

If we stack the num feature maps, we can treat the output of the convolution as another 3D volume/ image with C<sub>o</sub> = num channels.

In summary, the input to the convolutional layer is a volume with dimensions N<sub>i</sub> × N<sub>i</sub> × C<sub>i</sub> and the output is a volume of size N<sub>o</sub> × N<sub>o</sub> × num. Figure 2 shows a graphical picture.

### Pooling layer

A pooling layer is generally used after a convolutional layer to reduce the size of the feature maps. The pooling layer operates on each feature map separately and replaces a local region of the feature map with some aggregating statistic like max or average. In addition to reducing the size of the feature maps, it also makes the network invariant to small translations. This means that the output of the layer doesn’t change when the object moves a little.

In this project we will use only a MAX pooling layer shown in figure 3. This operation is performed in the same fashion as a convolution, but instead of applying a filter, we find the max value in each kernel. Let k represent the kernel size, s represent the stride and p represent the padding. Then the output of a pooling function f applied to a padded feature map X is given by:

<p align="center">
<img width="354" alt="image" src="https://user-images.githubusercontent.com/52765136/189805963-59412d77-1fc8-42a4-af0d-d119d0a1d5a6.png">
</p>

<p align="center">
<img width="388" alt="image" src="https://user-images.githubusercontent.com/52765136/189806029-5daf5338-6dc3-4fbe-a4fc-9807bda7884b.png">
</p>

### Activation layer - ReLU - Rectified Linear Unit
Activation layers introduce the non-linearity in the network and give the power to learn complex functions. The most commonly used non-linear function is the ReLU function defined as follows,

<p align="center">
f (x) = max(x, 0)
</p>

The ReLU function operates on each output of the previous layer.

### Loss layer
The loss layer has a fully connected layer with the same number of neurons as the number of classes. And then to convert the output to a probability score, a softmax function is used. This operation is given by,

<p align="center">
p = softmax(W x + b)
</p>

where, W is of size C × n where n is the dimensionality of the previous layer and C is the number of classes in the problem.

This layer also computes a loss function which is to be minimized in the training process. The most common loss functions used in practice are cross entropy and negative log-likelihood. In this project, we will just minimize the negative log probability of the given label.

### Architecture

In this project we will use a simple architecture based on a very popular network called the [LeNet](http://ieeexplore.ieee.org/abstract/document/726791/)

* Input - 1×28×28

* Convolution - k = 5, s = 1, p = 0, 20 filters

* ReLU

* MAXPooling - k=2, s=2, p=0

* Convolution - k = 5, s = 1, p = 0, 50 filters

* ReLU

* MAXPooling - k=2, s=2, p=0

* Fully Connected layer - 500 neurons

* ReLU

* Loss layer

Note that all types of deep networks use non-linear activation functions for their hidden layers. If we use a linear activation function, then the hidden layers has no effect on the final results, which would become the linear (affine) functions of the input values, which can be represented by a simple 2 layer neural network without hidden layers.

There are a lot of standard Convolutional Neural Network architectures used in the literature, for instance, AlexNet, VGG-16, or GoogLeNet. They are different in the number of parameters and their configurations.

### Data structures

We define four main data structures to help us implement the Convolutional Neural Network which are explained in the following section. Each layer is defined by a data structure, where the field type determines the type of the layer. This field can take the values of DATA, CONV, POOLING, IP, RELU, LOSS which correspond to data, convolution, max-pooling layers, inner-product/ fully connected, ReLU and Loss layers respectively. The fields in each of the layer will depend on the type of layer.

The input is passed to each layer in a structure with the following fields.

* height - height of the feature maps

* width - width of the feature maps

* channel - number of channels / feature maps

* batch size - batch size of the network.

In this implementation, we will implement the mini-batch stochastic gradient descent to train the network. The idea behind this is very simple, instead of computing gradients and updating the parameters after each image, we doing after looking at a batch of images. This parameter batch size determines how many images it looks at once before updating the parameters.

* data - stores the actual data being passed between the layers.

This is always supposed to be of the size [ height × width × channel, batch size ]. We can resize this structure during computations, but make sure to revert it to a two-dimensional matrix. The data is stored in a column major order. The row comes next, and the channel comes the last.

* diff - Stores the gradients with respect to the data, it has the same size as data. Each layer’s parameters are stored in a structure param. We do not touch this in the forward pass.

* w - weight matrix of the layer

* b - bias

param_grad is used to store the gradients coupled at each layer with the following properties:							

* w - stores the gradient of the loss with respect to w.

* b - stores the gradient of the loss with respect to the bias term.

### Forward Pass

Now we will start implementing the forward pass of the network. Each layer has a very similar prototype. Each layer’s forward function takes input, layer, param as argument. The input stores the input data and information about its shape and size. The layer stores the specifications of the layer (e.g., for a conv layer, it will have k, s, p). The params is an optional argument passed to layers which have weights. This contains the weights and biases to be used to compute the output. In every forward pass function, we are expected to use the arguments and compute the output. We are supposed to fill in the height, width, channel, batch size, data fields of the output before returning from the function. Also we should make sure that the data field has been reshaped to a 2D matrix.

#### Inner Product Layer

The inner product layer of the fully connected layer is implemented with the following definition

<p align="center">
[output] = inner_product_forward(input, layer, param)
</p>

#### Pooling Layer

We write a function which implements the pooling layer with the following definition.

<p align="center">
[output] = pooling_layer_forward(input, layer)
</p>

input and output are the structures which have data and the layer structure has the parameters specific to the layer. This layer has the following fields:
					
* pad - padding to be done to the input layer

* stride - stride of the layer

* k - size of the kernel (We assume square kernel)

#### Convolution Layer

Now we implement a convolution layer with the following definition.

<p align="center">
[output] = conv_layer_forward(input, layer, param)
</p>

The layer for a convolutional layer has the same fields as that of a pooling layer and param has the weights corresponding to the layer.

#### ReLu

Now we implement the ReLU function with the following definition.

<p align="center">
[output] = relu_forward(input)
</p>

### Back Propagation

After implementing the forward propagation, we will implement the back propagation using the chain rule. Let us assume layer i computes a function f<sub>i</sub> with parameters of w<sub>i</sub> then final loss can be written as the following.

<p align="center">
<img width="201" alt="image" src="https://user-images.githubusercontent.com/52765136/189810103-ecf99c9f-0e4f-455c-b6e8-80f0d41ab2da.png">
</p>

To update the parameters we need to compute the gradient of the loss w.r.t. to each of the parameters.

<p align="center">
<img width="122" alt="image" src="https://user-images.githubusercontent.com/52765136/189810221-f46dc65b-d36b-4c76-b87c-6245aa792c99.png">
</p>

where, h<sub>i</sub> = f<sub>i</sub>(w<sub>i</sub>, h<sub>i</sub>−1).

Each layer’s back propagation function takes input, output, layer, param as input and return param_grad and input_od. output.diff stores the <img width="25" alt="image" src="https://user-images.githubusercontent.com/52765136/189810370-cc71e49e-2947-46cd-9109-319bbaaa7c73.png">. We are to use this to compute <img width="25" alt="image" src="https://user-images.githubusercontent.com/52765136/189810711-512eaac4-61a7-4493-99fe-e673667e813e.png">
and store it in param_grad.w and  <img width="18" alt="image" src="https://user-images.githubusercontent.com/52765136/189810790-a584d8cd-40b9-45d9-9ca7-70c0493807e7.png"> to be stored in param_grad.b. We are also expected to return <img width="40" alt="image" src="https://user-images.githubusercontent.com/52765136/189810836-1ff41214-eed2-4826-b45a-f3af92a2462f.png"> in input_od, which is the gradient of the loss w.r.t the input layer.

#### ReLu

Now we implement the backward pass for the Relu layer in relu_backward.m file.

#### Inner Product Layer

We implement the backward pass for the Inner product layer in inner_product_backward.m.

#### Putting the network together

The function convnet forward takes care of this. This function takes the parameters, layers and input data and generates the outputs at each layer of the network. It also returns the probabilities of the image belonging to each class. In this function we can see how the data is being passed to perform the forward pass.

### Training

The function conv_net puts both the forward and backward passes together and trains the network.

<p align="center">
<img width="342" alt="image" src="https://user-images.githubusercontent.com/52765136/189811754-e51a2553-dab1-43e5-a399-992a0f894662.png">
</p>

#### Training

The script train_lenet.m defines the optimization parameters and performs the actual updates on the network. This script loads a pretrained network and trains the network for 3000 iterations. After training it for 3000 more iterations and saving the refined network weights as lenet.mat in the same format as lenet_pretrained.mat, the accuracy gets above 95%.

Summary of runs:
[First run 3000.pdf](https://github.com/Daneshpajouh/Digit-recognition-using-ConvNets/files/9559885/First.run.3000.pdf),
[Second Run 6000.pdf](https://github.com/Daneshpajouh/Digit-recognition-using-ConvNets/files/9559887/Second.Run.6000.pdf),
[Third run 5000.pdf](https://github.com/Daneshpajouh/Digit-recognition-using-ConvNets/files/9559890/Third.run.5000.pdf)


#### Test the network

The script test_network.m runs the test data through the network and obtains the prediction probabilities. This script generates the confusion matrix.

<p align="center">
<img width="550" alt="image" src="https://user-images.githubusercontent.com/52765136/189982795-423384a0-edb7-46c4-bafa-e3918e43a5d3.jpg">
</p>

<p align="center">
Figure 5: Confusion Matrix
</p>


### Visualization

We write a script vis_data.m which can load a sample image from the data, visualize the output of the second and third layers (i.e., CONV layer and ReLU layer). We show 20 images from each layer on a single figure file (like in Fig 4). To clarify, we take one image, run through our network, and visualize 20 features of that image at CONV layer and ReLU layer.

<p align="center">
Figure 6: Input image
</p>
<p align="center">
<img width="250" alt="image" src="https://user-images.githubusercontent.com/52765136/189984974-dc1885a6-f4db-4bc4-882c-dfa93c3c3c1d.jpg">
</p>

<p align="center">
Figure 7
</p>
<p align="center">
<img width="400" alt="image" src="https://user-images.githubusercontent.com/52765136/189985446-5a5aec84-866c-42a3-b390-9f12a5f8d080.jpg">
</p>


<p align="center">
Figure 8
</p>
<p align="center">
<img width="400" alt="image" src="https://user-images.githubusercontent.com/52765136/189985570-45333de0-fcb6-4629-ab16-da7cfec88e8e.jpg">
</p>

We will now try to use the fully trained network to perform the task of Optical Character Recognition. We use a set of real world images in the images folder. We write a script ec.m which will read these images and recognize the handwritten numbers.

The network we trained requires a grey-scale image with a single digit in each image. There are many ways to obtain this given a real image. Here is an outline of a possible approach:
  1.	Classify each pixel as foreground or background pixel by performing simple operations like thresholding.
  2.	Find connected components and place a bounding box around each character. We can use a matlab built-in function to do this.
  3.	Take each bounding box, pad it if necessary and resize it to 28×28 and pass it through the network.
  
<p align="center">
Figure 9
</p>
<p align="center">
<img width="400" alt="image" src="https://user-images.githubusercontent.com/52765136/189986468-2baea8ab-275d-4536-9440-c9a90f34e2e8.jpg">

</p>

<p align="center">
Figure 10
</p>
<p align="center">
<img width="400" alt="image" src="https://user-images.githubusercontent.com/52765136/189986620-2730b5b5-4364-49ac-9762-a3ac1e16a735.jpg">
</p>

<p align="center">
Figure 11
</p>
<p align="center">
<img width="400" alt="image" src="https://user-images.githubusercontent.com/52765136/189986677-9423b7b1-85aa-4ee7-b90a-b62c2b0c54f7.jpg">
</p>

There might be errors in the recognition. We can use graythresh, adaptthresh, bwconncomp, bwlabel, and regionprops built-in functions.

#### Appendix: List of all files in the project

* col2im_conv.m - Helper function, we can use this if needed

* col2im_conv_matlab.m - Helper function, we can use this if needed

* conv_layer_backward.m

* conv_layer forward.m

* conv_net.m

* convnet_forward.m

* get_lenet.m - Has the architecture.

* get_lr.m - Gets the learning rate at each iterations

* im2col_conv.m - Helper function, we can use this if needed

* im2col_conv_matlab.m - Helper function, we can use this if needed

* init_convnet.m - Initialise the network weights

* inner_product_backward.m

* inner_product_forward.m

* load_mnist.m - Loads the training data.

* mlrloss.m - Implements the loss layer

* pooling_layer_backward.m

* pooling_layer_forward.m

* relu_backward.m

* relu_forward.m

* sgd_momentum.m - Has the update equations

* test_network.m - Test script

* train_lenet.m - Train script

* vis_data.m - Visualises the filters

* lenet_pretrained.mat - Trained weights

* mnist_all.mat - Dataset
