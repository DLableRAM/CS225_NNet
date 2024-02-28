#include <iostream>
#include <cmath>

#include "defs.cuh"

class layer {
  private:
    int size;
    float* weights;
    float* bias;
  public:
    int getSize();
    void getWeights(float* out);
    void getBias(float* out);
    void setWeights(float* in) { weights = in; }
    void setBias(float* in) { bias = in; }
    layer(int s): size(s) {
      weights = new float[size];
      bias = new float[size];
    }
    ~layer() {
      //free allocated memory
      delete weights;
      delete bias;
    }
};

class neuralnet {
  private:
   //layercount
   //layer array
   //xlayer (temporary intermediate layer)
   void dotProduct(float* input1, float* input2, float* output, int size);
   void activation(float* output, float* input, float* bias, int size);
  public:
    void inference(float* output, float* input);
    //train()
};

void neuralnet::dotProduct(float* input1, float* input2, float* output, int size) {
  //invoke kernel
  dotProduct<<<numBlocks, numThreads>>>(input1, input2, output);
}

void neuralnet::activation(float* output, float* input, float* bias, int size) {
  //sigmoid activation function, also adds bias
  float z[size];
  int i;
  for (i = 0; i < size; ++i) {
    z[i] = input[i] + bias[i];
    output[i] = 1.0/(1.0 + std::exp(-z[i]));
  }
}
