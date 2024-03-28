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

void neuralnet::getOutput(float* out) {
  for (int i = 0; i < outputSize; ++i) {
    out[i] = output[i];
  }
}

void neuralnet::setInput(float* in) {
  for (int i = 0; i < inputSize; ++i) {
    input[i] = in[i];
  }
  //write input to vram
  int inputDataSize = inputSize*sizeof(float);
  cudaMemcpy(device_input, input, inputDataSize, cudaMemcpyHostToDevice);
}

void neuralnet::infer() {
  //Copy input to vram
  int inputDataSize = inputSize*sizeof(float);
  cudaMemcpy(device_input, input, inputDataSize, cudaMemcpyHostToDevice);
  //call gpu kernel
  inference<<<numBlocks, numThreads>>>(device_input, inputSize, device_output, outputSize, device_wmatrix, device_bias, hiddenLayerCount, hiddenLayerSize);
  cudaDeviceSynchronize();
  int outputDataSize = (hiddenLayerSize*hiddenLayerCount + outputSize)*sizeof(float);
  cudaMemcpy(device_output, output, outputDataSize, cudaMemcpyDeviceToHost);
}

void neuralnet::trn(float lr, int epochs, std::string directory) {
  //load dataset from files
  //files must be named numbers from 0 to whatever you want as long as
  //they are continuous ints.
  //they also MUST be formatted as input vector followed by output vector on newlines
  std::ifstream fileread;
  int filecount = 0;
  bool validfile;
  std::string fulldir;
  std::cout<<"Scanning valid files..."<<std::endl;
  do {
    fulldir = /*directory + "/" + */std::to_string(filecount);
    fileread.open(fulldir);
    validfile = false;
    if (fileread.good()) {
      ++filecount;
      validfile = true;
    }
    fileread.close();
  } while(validfile);
  std::cout<<filecount<<" valid files found."<<std::endl;
  //begin training loop
  for (epochs = epochs; epochs > 0; --epochs) {
  for (int j = 0; j < filecount; ++j) {
    fulldir = directory + "/" + std::to_string(j);
    fileread.open(fulldir);
    //load input to vram
    float inputbuffer[inputSize];
    for (int i = 0; i < inputSize; ++i) {
      fileread >> inputbuffer[i];
    }
    //inference
    setInput(inputbuffer);
    infer();
    //get error
    float er;
    float result[outputSize];
    float prediction[outputSize];
    for (int i = 0; i < outputSize; ++i) {
      result[i] = output[hiddenLayerCount*hiddenLayerSize + i];
    }
    for (int i = 0; i < outputSize; ++i) {
      fileread >> prediction[i];
    }
    fileread.close();
    for (int i = 0; i < outputSize; ++i) {
      er += (result[i] - prediction[i]);
    }
    er = (2.0/outputSize)*er;
    std::cout<<"Current error is:"<<er<<std::endl;
    //call kernel
    train<<<numBlocks, numThreads>>>(device_input, lr, device_output, inputSize, er, outputSize, device_wmatrix, device_bias, hiddenLayerCount, hiddenLayerSize);
    cudaDeviceSynchronize();
  }
  }
}

int neuralnet::getInputSize() {
  return inputSize;
}

int neuralnet::getOutputSize() {
  return outputSize;
}

//operator overloading
std::ostream& operator<< (std::ostream& os, const neuralnet& n) {
  os << n.name;
  return os;
}

