# CS225_NNet
A CUDA-based neural network from scratch, class project. Because it is a class project there are some shoehorned things (like overloading the insertion operator) required. Of course it went very far out-of-scope for a class project, but it was fun anyway.

To give a rundown on the network's architecture, it is a simple linear regression based network with a defined input vector size, output vector size, hidden layer width and hidden layer depth. It can forward and backward propogate.

It's far from practical but feel free to borrow or do whatever with it as a toy example of GPU computation and neural networks. It cannot save or load any weights.

## How to compile
You'll need cuda/nvcc and cmake, you just run the makefile. The executable will be named NNet.
