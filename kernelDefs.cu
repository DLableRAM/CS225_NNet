#include "defs.cuh"

__global__ void inference(float* input, int inputSize, float* output, int outputSize, float*** weights, float** bias, int layercount, int layerwidth) {
  //Create row/column indicies
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  //Create temporary vector for intermediate values
  float temp[layerwidth];
    
  //Perform first matrix multiplication
  if (row < layerwidth && col < inputSize) {
      float sum[layerwidth];
      for (int i = 0; i < inputSize; ++i) {
          sum[row] += input[i] * weights[0][row][i];
      }
      temp[row] = sum[row];
  }

  //Add bias to first multiplication
  temp[row] += bias[row][0];
  
  //Sigmoid activation function
  temp[row] = 1.0/(1.0 + expf(-temp[row]));

  //Iterate hidden layers
  if (row < layerwidth) {
    for (int j = 1; j < (layercount-1); ++j) {
      float sum[layerwidth];
      for (int i = 0; i < layerwidth; ++i) {
        sum[row] += temp[i] * weights[j][row][i];
      }
      temp[row] = sum[row];
      //Add bias
      temp[row] += bias[row][j];
      //Sigmoid activation function
      temp[row] = 1.0/(1.0 + expf(-temp[row]));
    }
  }

  //The final weight matrix needs to adapt to the output neurons,
  //similar to the input of arbitrary size.
  if (row < layerwidth && col < inputSize) {
      float sum[layerwidth];
      for (int i = 0; i < outputSize; ++i) {
          sum[row] += temp[i] * weights[layercount][row][i];
      }
      output[row] = sum[row];
  }

  //Add bias to final multiplication
  output[row] += bias[row][layercount];
  
  //Sigmoid activation function
  output[row] = 1.0/(1.0 + expf(-output[row]));
}

//This is an alternative version of the inference kernel which returns the full sigmoid output instead of just the output. It is used for backpropagation.
__global__ void fullSigmoid(float* input, int inputSize, float** output, int outputSize, float*** weights, float** bias, int layercount, int layerwidth) {
  //Create row/column indicies
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
    
  //Perform first matrix multiplication
  if (row < layerwidth && col < inputSize) {
      float sum[layerwidth];
      for (int i = 0; i < inputSize; ++i) {
          sum[row] += input[i] * weights[0][row][i];
      }
      output[0][row] = sum[row];
  }

  //Add bias to first multiplication
  output[0][row] += bias[row][0];
  
  //Sigmoid activation function
  output[0][row] = 1.0/(1.0 + expf(-output[0][row]));

  //Iterate hidden layers
  if (row < layerwidth) {
    for (int j = 1; j < (layercount-1); ++j) {
      float sum[layerwidth];
      for (int i = 0; i < layerwidth; ++i) {
        sum[row] += output[j-1][i] * weights[j][row][i];
      }
      output[j][row] = sum[row];
      //Add bias
      output[j][row] += bias[row][j];
      //Sigmoid activation function
      output[j][row] = 1.0/(1.0 + expf(-output[j][row]));
    }
  }

  //The final weight matrix needs to adapt to the output neurons,
  //similar to the input of arbitrary size.
  if (row < layerwidth && col < inputSize) {
      float sum[layerwidth];
      for (int i = 0; i < outputSize; ++i) {
          sum[row] += output[layercount-1][i] * weights[layercount][row][i];
      }
      output[layercount][row] = sum[row];
  }

  //Add bias to final multiplication
  output[layercount][row] += bias[row][layercount];
  
  //Sigmoid activation function
  output[layercount][row] = 1.0/(1.0 + expf(-output[layercount][row]));
}

//I may just end up doing error calcs on the CPU because the timesave probably isn't worth it.
__global__ void error(float* predicted, float* actual, float* output, int outputSize) {
  int thread = threadIdx.x;
  float temp[outputSize];
  if (thread < outputSize) {
  temp[thread] = actual[thread] - predicted[thread];
  }
  for (int i = 0; i < outputSize; ++i) {
    output += temp[i];
  }
  output = output/outputSize;
}

//Train one iteration. Might pre-determine iterations to save kernel calls?
__global__ void train(float* input, float learnrate, float** sigmoid, int inputSize, float* prediction, float error, int outputSize, float*** weights, float** bias, int layercount, int layerwidth) {
  int threadx = threadIdx.x;
  int thready = threadIdx.y;
  //the first set of x values for training is the input vector. After that, it is the output vector of the previous layer.
  if (threadx < layerwidth && thready < inputSize) {
    weights[0][threadx][thready] -= learnrate * error * sigmoid[0][threadx] * (1 - sigmoid[0][threadx]) * input[thready];
    bias[0][threadx] -= learnrate * error * sigmoid[0][threadx] * (1 - sigmoid[0][threadx]);
  }
  //TODO: parallelize better
  for (int i = 0; i < (layercount - 1); ++i) {
    if (threadx < layerwidth && thready < layerwidth) {
      weights[i][threadx][thready] -= learnrate * error * sigmoid[i][threadx] * (1 - sigmoid[i][threadx]) * sigmoid[i-1][threadx];
      bias[i][threadx] -= learnrate * error * sigmoid[i][threadx] * (1 - sigmoid[i][threadx]);
    }
  }
  if (threadx < layerwidth && thready < outputSize) {
    weights[layercount][threadx][thready] -= learnrate * error * sigmoid[layercount][threadx] * (1 - sigmoid[layercount][threadx]) * input[thready];
    bias[layercount][threadx] -= learnrate * error * sigmoid[layercount][threadx] * (1 - sigmoid[layercount][threadx]);
  }
}
