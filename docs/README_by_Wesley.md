### Overview of Code 
- Unet is currently the most widely used image (semantic) segmentation model. It adopts the structure of encode (encoding) + decode (decoding). First, it performs multiple conv (+Bn+Relu) + pooling downsampling on the image, and then upsamples the low-level feature map before crop, and after upsampling The feature map is fused, and the upsampling + fusion process is repeated until a segmentation map with the same size as the input image is obtained.

- ResNet is to solve the deep neural network "degradation" problem. As we know, the performance of layers, model on the training set and the test set will be better, because the 
model complexity is higher, the expression ability is stronger, and the potential mapping relation can be fitted better. Degradation refers to the rapid decline of performance after adding more layers to the network.

- DenseNet can ensure the maximum degree of information transmission between the middle layers of the network, directly connect to all layers

### dataset 

- The dataset I am using is from http://www.vcip2020.org/grand_challenge.htm
- Download the dataset and place the folder under ./data/


- Resnet+Unet code
- DenseNet Code
- Added DenseNet for alternative comparison with the proposed method

### How to run:
- Open the folder in VSCode and launch the launch.json to run the training program
- Add the data souce file under /data/ folder
- Adjust the correlated folder paths 
- You may see the training process in local host by launching Visdom. 


