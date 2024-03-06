__global__ void inference(float* input, int inputSize, float* output, int outputSize, float*** weights, float** bias, int layercount, int layerwidth) {
  //Create row/column indicies
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  //Create temporary vector for intermediate values
  float temp[layerwidth];
    
  //Perform first matrix multiplication
  if (row < layerwidth && col < inputSize) {
      float sum[row];
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
      float sum[row];
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
      float sum[row];
      for (int i = 0; i < outputSize; ++i) {
          sum[row] += temp[i] * weights[layercount][row][i];
      }
      output[row] = sum[row];
  }

  //Add bias to first multiplication
  output[row] += bias[row][0];
  
  //Sigmoid activation function
  output[row] = 1.0/(1.0 + expf(-temp[row]));
}
