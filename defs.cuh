#pragma once

#include <fstream>
#include <iterator>
#include <system_error>
#include <iostream>

//static defs
static const dim3 numBlocks(512,1,1);
static const dim3 numThreads(512,1,1);

//kernel functions
__global__ void inference();
__global__ void train();

//class defs
class layer {
  private:
    int inputSize;
    int outputSize;
    float** weightMatrix;
    float* bias;
  public: 
    //create new layer, defaults to square which is typical for hidden layers
    layer(int i = 0, int o = 0): inputSize(i), outputSize(o) { }
    //create layer with values (IE from file)
    layer(int i, int o, float** wm, float* b): inputSize(i), outputSize(o), weightMatrix(wm), bias(b) { }
    //allocate layer memory
    void allocLayer() {
      weightMatrix = new float[outputSize][inputSize];
      bias = new float[outputSize];
    }
    //set input/output size, this is mainly to configure the endpoint layers.
    void setInputSize(int in) { inputSize = in; }
    void setOutputSize(int in) { outputSize = in; }
    //free allocated memory
    ~layer() {
      for (int i = 0; i < outputSize; ++i) {
        delete [] weights[i];
      }
      delete [] weights;
      delete [] bias;
    }
};

class neuralnet {
  private:
    //fields
    std::string name;
    int inputSize;
    int outputSize;
    int hiddenLayerCount;
    int hiddenLayerSize;
    float* input;
    float* output;
    layer* hiddenlayers;
    //methods
    //export weights into degree 3 tensor
    //This is needed to do a cuda memcpy
    float*** exportWeights(layer* layers);
    //export biases into matrix, same reason
    float** exportBias(layer* layers);
  public:
    //methods
    //construct new neuralnet
    neuralnet(int ins, int ops, int hls, int hlc, std::string n);
    //load neuralnet from file
    neuralnet(std::ifstream& nnet);
    //clear the allocated memory
    ~neuralnet();
    //inference network
    void inference(float* input, float* output);
    //train network from file
    void train(std::fstream& dataset);
    //save network to file
    void save(std::string path);
    //set input
    void setInput(float* in) { input = in; }
    //get input, in case you forgot!
    float* getInput() const { return input; }
    //get output, there is no setter because it is an output
    //hence the name...
    float* getOutput() const { return output; }
};
