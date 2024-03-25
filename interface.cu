#include "defs.cuh"

void ui::run() {
  std::string userinpt;
  nnetUserCreate(); 
  while (userinpt != "exit") {
    std::cout<<"This is a test. Type 'exit' to leave."<<std::endl;
    getstring(userinpt);
  }
  //clear this memory before shutdown.
  delete user_nnet;
}

void ui::nnetUserCreate() {
  std::string n;
  int ins;
  int ops;
  int hls;
  int hlc;
  std::cout<<"Welcome to the simple CUDA neural net manager! Time to create a neural net..."<<std::endl
    <<"Name the neural net: ";
  getstring(n);
  std::cout<<"Define input size: ";
  getint(ins);
  std::cout<<"Define output size: ";
  getint(ops);
  std::cout<<"Define hiddenlayer size: ";
  getint(hls);
  std::cout<<"Define hiddenlayer count: ";
  getint(hlc);

  user_nnet = new neuralnet(ins, ops, hls, hlc, n);
  std::cout<<"Neural network created. Loading into vram..."<<std::endl;

  user_nnet->loadNet();
}
