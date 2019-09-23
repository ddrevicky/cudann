# cudann
This is a simple framework implementing training and inference of fully-connected neural networks. Two implementations are provided: CPU version uses [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) for matrix operations and GPU version uses CUDA.

## Details
### General forward and backward propagation model
The computation model in the framework is based on layers (affine, ReLU, sigmoid, etc.) each of which implements a forward and backward interface. The forward pass takes the input data (generally a tensor), performs some computation and outputs the transformed data. Layers can then be concatenated one after another with each layer taking the previous one’s output as input, transforming it and sending it to the next layer in the chain. The final layer in the chain typically implements a loss function (e.g. cross-entropy). 

![](https://github.com/ddrevicky/cudann/blob/master/imgs/forward_backward.png)

For the network to learn, the parameters in the model have to be adjusted to decrease the value of the loss function using backpropagation of gradients. Firstly, the gradient of the loss function is computed with respect to the last layer in the chain. This module then passes this gradient in the
backward pass to the previous layers until the loss gradient of each trainable parameter is computed. See Fig 1. We then perform the parameter update during which the weights are slightly shifted in the direction which decreases the loss.

### CPU and GPU version comparison
This modular approach allows the CPU and GPU versions of the neural network to be fairly similar. Both consist of a user-defined sequence of layers(modules) which modify the data that flows through the network in the forward and backward pass. The greatest difference is that the data is allocated on the GPU for the accelerated version. The forward and backward methods of the GPU layers also end up calling a CUDA kernel which operates on the data instead of processing it on the CPU.

### GPU implementation details
#### GPU memory allocation
For the purpose of training all the necessary memory is allocated on the GPU just once at the beginning. This includes the memory required for the model parameters, parameter gradients, input data and activations computed at each layer. If the user does not request information about the training loss, no memory is copied from the device to host until the end of the final training epoch.

#### Asynchronous batch transfer
Instead of allocating enough space just for a single training batch, the framework allocates two memory pools: a *computing pool* and a *copying pool* each having the size of multiple data batches. At the beginning of an epoch, the first few batches are copied to one of the memory pools synchronously. This pool is designated as the current computing pool and training on it immediately begins. At the same time but on a different stream, an asynchronous copy of next couple of batches starts with the currently unused memory pool (now designated as the copying pool) as their destination. Once the training stream finishes processing all of the batches in the computing pool, the computing and copying pools are swapped. Assuming that the copying has finished (the computing stream waits for an CUDA event which guarantees that) the new computing pool is now used for training and an asynchronous copy is started in the new copying pool. This process is repeated until the end of the training epoch.

The memory pools have the size of multiple batches (and not just a single one) because the overhead for initiating a memory transfer from host to device is much larger than the one for copying several batches at once instead of just one.

## Evaluation
The performance of the CPU and GPU version of the framework was compared for several different network architectures and for varying batch sizes below:

### Architecture with five 300 neuron hidden layers
![](https://github.com/ddrevicky/cudann/blob/master/imgs/table2.png)

### Architecture with nine 1000 neuron hidden layers
![](https://github.com/ddrevicky/cudann/blob/master/imgs/table1.png)

Both comparisons were computed on a GeForce GTX 970M, Intel Core i5-4210H, 8GB RAM system. Note that this is quite an old card so the GPU speedup would be much higher if evaluated on something like GTX 1080 Ti.

## Acknowledgements
The MNIST dataset is loaded using a [library](https://github.com/wichtounet/mnist) created by Baptiste Wicht and I've also used Taylor C. Richtberger's [library](https://github.com/Taywee/args) for argument parsing. Thanks!
