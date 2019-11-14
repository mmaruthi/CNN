# CNN
Convolution Neural Networks

Definitions:

1.Convolution : Convolution is a process in which we amplify the image/signal/sensor etc to make/view it bigger for our reference. Say suppose if we have an input image or signal , we will convolve with a kernel or feature extractor so that the features of the image can be amplified and extracted.

2.Filters / Kernels : Filters or kernels are the convolution matrix(matrices) we use to convolve to extract features from the input image

3.Epoch : Executing our Neural network model one time through our entire data base of images is called Epoch. Eg : Say , My data base has 50 images for which i have to do Image classification , then if my Neural network model executes on all the 50 images once i.e it completed one epoch. We will give how many epochs we want to run while we are fitting our model.

4. 1 * 1 Convolution : Means we use a Kernel/Matrix size of 1 * 1 for convolution operation . We use it in transition layer to link the channels together 

5. 3 * 3 convolution : Means we use a kernel/Matrix size of 3 * 3 for convolution operation. This is the most commonly used filter size to extract the features. This is most commonly used because of its axis symmnetry.

6. Feature Maps : The area or output deduced or received from convolving an image with a kernel or matrix is called Feature map. Say a 300 * 300 image is convolved by 3* 3 matrix to extract its features , we get an output of 298* 298. This 298 * 298 is called a Feature Map here.

7. Activation function : Act function decides which neuron to get activated when and of how much level it should be activated .Our brain has some neurons for taste ,smell etc etc which gets activated as per the input (smell, taste).same is the case with activation function. Some activation functions are Sigmoid, Tan H , Relu , softmax etc.

8. Receptive Field : Area through which a filter can see the image at a time is the receptive field .If seen through middle layers that becomes its corresponding lcoal receptive field . If seen through the end layer , that becomes the gloabl receptive field.

