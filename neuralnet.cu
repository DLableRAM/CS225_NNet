#include "defs.cuh"

neuralnet::neuralnet(int ins, int ops, int hls, int hlc, std::string n) {
  name = n;
  inputSize = ins;
  outputSize = ops;
  hiddenLayerSize = hls;
  hiddenLayerCount = hlc;
  input = new float[inputSize];
  output = new float[hiddenLayerSize*hiddenLayerCount + outputSize];
  wmatrix = new float[hiddenLayerSize*hiddenLayerSize*hiddenLayerCount + inputSize*hiddenLayerSize + outputSize*hiddenLayerSize];
  bias = new float[hiddenLayerSize*hiddenLayerCount + outputSize];
}

neuralnet::~neuralnet() {
  delete [] input;
  delete [] output;
  delete [] wmatrix;
  delete [] bias;
  cudaFree(device_input);
  cudaFree(device_output);
  cudaFree(device_wmatrix);
  cudaFree(device_bias);
}

void neuralnet::loadNet() {
  //TODO: Add a safeguard to prevent loading models multiple times.
  int inputDataSize = inputSize*sizeof(float);
  int outputDataSize = (hiddenLayerSize*hiddenLayerCount + outputSize)*sizeof(float);
  int wmatrixDataSize = (hiddenLayerSize*hiddenLayerSize*hiddenLayerCount + inputSize*hiddenLayerSize + outputSize*hiddenLayerSize)*sizeof(float);
  int biasDataSize = (hiddenLayerSize*hiddenLayerCount + outputSize)*sizeof(float);

  //TODO: Add malloc error checking.
  cudaMalloc(&device_input, inputDataSize);
  cudaMalloc(&device_output, outputDataSize);
  cudaMalloc(&device_wmatrix, wmatrixDataSize);
  cudaMalloc(&device_bias, biasDataSize);
  
  if ((device_input == NULL) || (device_output == NULL) || (device_wmatrix == NULL) || (device_bias == NULL)) {
    errmsg = "Failed to allocate memory in VRAM.";
    throw(errmsg);
  }

  cudaMemcpy(device_input, input, inputDataSize, cudaMemcpyHostToDevice);
  cudaMemcpy(device_output, output, outputDataSize, cudaMemcpyHostToDevice);
  cudaMemcpy(device_wmatrix, wmatrix, wmatrixDataSize, cudaMemcpyHostToDevice);
  cudaMemcpy(device_bias, bias, biasDataSize, cudaMemcpyHostToDevice);
}

float* neuralnet::getInput() {
  //I don't think I should cudaMemcpy here?
  return input;
}

void neuralnet::getOutput(float* out) {
  //Pulls output from vram
  int outputDataSize = (hiddenLayerSize*hiddenLayerCount + outputSize)*sizeof(float);
  cudaMemcpy(device_output, output, outputDataSize, cudaMemcpyDeviceToHost);
  for (int i = 0; i < outputSize; ++i) {
    out[i] = output[i];
  }
}

void neuralnet::infer() {
  //Copy input to vram
  int inputDataSize = inputSize*sizeof(float);
  cudaMemcpy(device_input, input, inputDataSize, cudaMemcpyHostToDevice);
  //call gpu kernel
  inference<<<numBlocks, numThreads>>>(device_input, inputSize, device_output, outputSize, device_wmatrix, device_bias, hiddenLayerCount, hiddenLayerSize);
  cudaDeviceSynchronize();
}

void neuralnet::trn(std::string directory, float lr) {
  //load dataset from file

  //iterate over file values

  //load input to vram

  //inference

  //get error
  float er;
  train<<<numBlocks, numThreads>>>(device_input, lr, device_output, inputSize, er, outputSize, device_wmatrix, device_bias, hiddenLayerCount, hiddenLayerSize);
}

//operator overloading
std::ostream& operator<< (std::ostream& os, const neuralnet& n) {
  os << n.name;
  return os;
}

