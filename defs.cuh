//enums
#include <system_error>
enum thread_params {
  numBlocks = 1,
  numThreads = 16
};

//kernel functions
__global__ void dotProduct();
__global__ void activationfunction();
