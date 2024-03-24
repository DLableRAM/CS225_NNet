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

//I know making global variables is bad. But this is just to store an output string in the event something goes wrong. Would rather just reuse this everywhere than allocate a ton of these.
static std::string errmsg;

//class defs
class neuralnet {
  private:
    //fields
    std::string name;
    int inputSize;
    int outputSize;
    int hiddenLayerCount;
    int hiddenLayerSize;
    float* wmatrix;
    float* bias;
    float* input;
    float* output;
    float* device_wmatrix;
    float* device_bias;
    float* device_input;
    float* device_output;
  public:
    //methods
    //construct new neuralnet
    neuralnet(int ins, int ops, int hls, int hlc, std::string n);
    //load neuralnet from file (possible feature if I have time)
    //neuralnet(std::ifstream& nnet);
    //clear the allocated memory
    ~neuralnet();
    //load into GPU
    void loadNet();
    //inference network,
    void inference();
    //train network from file
    void train(std::string directory);
    //save network to file
    void save(std::string path);
    //set input
    void setInput(float* in) { input = in; }
    //get input, in case you forgot!
    float* getInput() const { return input; }
    //get output, there is no setter because it is an output
    //hence the name...
    float* getOutput() const { return output; }
    //I really don't like friend functions but the assignment requires it.
    friend std::ostream& operator<< (std::ostream& os);
};

//and I'll just put the implementation right here since it's so short.
friend std::ostream& operator<< (std::ostream& os) {
  os << name;
  return os;
}

class inputhandler {
    public:
        void getint(int& x) {
            std::cin >> x;
            while (!std::cin) {
                std::cin.clear();
                std::cin.ignore(INT_MAX, '\n');
                std::cout << "Invalid input. Try again: ";
                std::cin >> x;
            }
        }

        void getfloat(float& x) {
            std::cin >> x;
            while (!std::cin) {
                std::cin.clear();
                std::cin.ignore(INT_MAX, '\n');
                std::cout << "Invalid input. Try again: ";
                std::cin >> x;
            }
        }

        void getstring(std::string& x) {
            std::cin >> x;
            return x;
        }
};

class ui : public inputhandler {
  private:
    neuralnet* user_nnet;
  public:
    //runs the user interface until user quits
    void run();
}
