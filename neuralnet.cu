#include "defs.cuh"

void neuralnet::dotProduct(float* input1, float* input2, float* output, int size) {
  //invoke kernel
  dotProduct<<<numThreads/size + 1, numThreads>>>(input1, input2, output, size);
}

void neuralnet::activation(float* input, float* bias, float* output, int size) {
  //sigmoid activation function, also adds bias
  activationfunction<<<numThreads/size + 1, numThreads>>>(input, bias, output, size);
}
