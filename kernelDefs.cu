#include "defs.cuh"

//NOTE: Originally, I was going to make 2d/3d matricies into arrays of arrays. However, this is stupid.
//2D array[layer_index*layerwidth + val_index]
//3D array[layer_index*layercount*layerwidth + neuron_index*layerwidth + weight_index]
//This SHOULD stay consistent

__global__ void inference(float* input, int inputSize, float* output, int outputSize, float* weights, float* bias, int layercount, int layerwidth) {
  //Create thread indicies
  //Each independent neuron gets a thread for it's calculations
  int neurons = threadIdx.x;

  //Perform first matrix multiplication
  if (neurons < layerwidth) {
      float sum[layerwidth];
      for (int i = 0; i < inputSize; ++i) {
          //at layer 0, we don't need to index those
          sum[neurons] += input[i] * weights[neurons*layerwidth + i];
      }
      output[neurons] = sum[neurons];
      //Add bias to first multiplication
      //since it is the first column, with index zero, we don't need column index
      output[neurons] += bias[neurons];
  
      //Sigmoid activation function
      output[neurons] = 1.0/(1.0 + expf(-output[neurons]));
  }

  //Iterate hidden layers
  if (neurons < layerwidth) {
    for (int j = 1; j < (layercount-1); ++j) {
      float sum[layerwidth];
      for (int i = 0; i < layerwidth; ++i) {
        sum[neurons] += output[(j-1)*layerwidth + i] * weights[j*layercount*layerwidth + neurons*layerwidth + i];
      }
      output[j*layerwidth + neurons] = sum[neurons];
      //Add bias
      output[j*layerwidth + neurons] += bias[j*layerwidth + neurons];
      //Sigmoid activation function
      output[j*layerwidth + neurons] = 1.0/(1.0 + expf(-output[j*layerwidth + neurons]));
    }
  }

  //The final weight matrix needs to adapt to the output neurons,
  //similar to the input of arbitrary size.
  if (neurons < layerwidth) {
      float sum[layerwidth];
      for (int i = 0; i < outputSize; ++i) {
          sum[neurons] += output[(layercount-1)*layerwidth + i] * weights[layercount*layercount*layerwidth + neurons*layerwidth + i];
      }
      output[layercount*layerwidth + neurons] = sum[neurons];
      //Add bias to final multiplication
      output[layercount*layerwidth + neurons] += bias[layercount*layerwidth + neurons];
  
      //Sigmoid activation function
      output[layercount*layerwidth + neurons] = 1.0/(1.0 + expf(-output[layercount*layerwidth + neurons]));
  }
}

//Train one iteration. Might pre-determine iterations to save kernel calls?
__global__ void train(float* input, float learnrate, float* sigmoid, int inputSize, float error, int outputSize, float* weights, float* bias, int layercount, int layerwidth) {
  //Since training is weight and neuron independent, we'll parallelize it that way.
  int neurons = threadIdx.x;
  int weightinx = threadIdx.y;
  //the first set of x values for training is the input vector. After that, it is the output vector of the previous layer.
  if (neurons < layerwidth && weightinx < inputSize) {
    weights[neurons*layerwidth + weightinx] -= learnrate * error * sigmoid[neurons] * (1 - sigmoid[neurons]) * input[neurons];
    bias[neurons] -= learnrate * error * sigmoid[neurons] * (1 - sigmoid[neurons]);
  }
  //TODO: parallelize better
  for (int i = 0; i < (layercount - 1); ++i) {
    if (neurons < layerwidth && weightinx < layerwidth) {
      weights[i*layercount*layerwidth + neurons*layerwidth + weightinx] -= learnrate * error * sigmoid[i*layerwidth + neurons] * (1 - sigmoid[i*layerwidth + neurons]) * sigmoid[(i-1)*layerwidth + neurons];
      bias[i*layerwidth + neurons] -= learnrate * error * sigmoid[i*layerwidth + neurons] * (1 - sigmoid[i*layerwidth + neurons]);
    }
  }
  if (neurons < layerwidth && weightinx < outputSize) {
    weights[layercount*layercount*layerwidth + neurons*layerwidth + weightinx] -= learnrate * error * sigmoid[layercount*layerwidth + neurons] * (1 - sigmoid[layercount*layerwidth + neurons]) * sigmoid[(layercount-1)*layerwidth + neurons];
    bias[layercount*layerwidth + neurons] -= learnrate * error * sigmoid[layercount*layerwidth + neurons] * (1 - sigmoid[layercount*layerwidth + neurons]);
  }
}
