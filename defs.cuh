#pragma once

#include <fstream>
#include <system_error>
#include <iostream>

//enums
enum thread_params {
  numBlocks = 1,
  numThreads = 256
};

//kernel functions
//replace these
__global__ void dotProduct();
__global__ void activationfunction();
//with this
__global__ void inference();
__global__ void train();

//class defs
class layer {
  private:
    int inputSize;
    int outputSize;
    float* weightMatrix;
    float* bias;
  public: 
    //create new layer, defaults to square which is typical for hidden layers
    layer(int i, int o): inputSize(i), outputSize(o) {
      weightMatrix = new float[outputSize][inputSize];
      bias = new float[outputSize];
    }
    //create layer with values (IE from file)
    layer(int i, int o, float* wm, float* b): inputSize(i), outputSize(o), weightMatrix(wm), bias(b) { }
    //free allocated memory
    ~layer() {
      delete weights;
      delete bias;
    }
};

class neuralnet {
  private:
    //fields
    int inputSize;
    int outputSize;
    int hiddenLayerCount;
    int hiddenLayerSize;
    float* input;
    float* output;
    layer* hiddenlayers;
    //methods
    //export weights into degree 3 tensor
    float* exportWeights(layer* layers);
    //export biases into matrix
    float* exportBias(layer* layers);

    //restructure these functions to do less kernel calls
    //since transferring data between host and device is slow
    //this will involve passing a 3 dimensional array to device memory
    void dotProduct(float* input1, float* input2, float* output, int size);
    void activation(float* input, float* bias, float* output, int size);
  public:
    //methods
    //construct new neuralnet
    neuralnet(int ins, int ops, int hls, int hlc);
    //load neuralnet from file
    neuralnet(std::fstream& nnet);
    //inference network
    void inference(float* input, float* output);
    //train network from file
    void train(std::fstream& dataset);
    //save network to file
    void save(std::string path);
    //set input
    void setInput(float* in) { input = in; }
    //get input, in case you forgot!
    const float* getInput() { return input; }
    //get output, there is no setter because it is an output
    //hence the name...
    const float* getOutput() { return output; }
};
