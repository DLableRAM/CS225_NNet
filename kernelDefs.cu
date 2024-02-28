__global__ void dotProduct(float* input1, float* input2, float* output, int size) {
  int i = threadIdx.x;
  if (i < size) {
    output[i] = input1[i]*input2[i];
  }
}

__global__ void activationfunction(float* input, float* bias, float* output, int size) {
  float z[size];
  int i = threadIdx.x;
  if (i < size) {
    z[i] = input[i] + bias[i];
    output[i] = 1.0/(1.0 + expf(-z[i]));
  }
}
