#include "defs.cuh"

void ui::run() {
  int userinpt = 1;
  nnetUserCreate();
  while (userinpt != 0) {
    std::cout<<"Neural net: "<<*user_nnet<<" is active."<<std::endl;
    std::cout<<"Input your command. Commands are: X Y Z"<<std::endl;
    getint(userinpt);
    mainmenu(userinpt);
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

  try {
    user_nnet->loadNet();
  }
  catch(std::string s) {
    std::cout<<s<<std::endl;
  }
}

void ui::mainmenu(int input) {
  float inBuffer[user_nnet->getInputSize()];
  float outBuffer[user_nnet->getOutputSize()];
  std::string tdir;
  float learnrate;
  int ep;
  switch (input) {
    case 0:
      //exit
      std::cout<<"Shutting down..."<<std::endl;
      break;
    case 1:
      //inference
      std::cout<<"Creating input vector..."<<std::endl;
      for (int i = 0; i < user_nnet->getInputSize(); ++i) {
        std::cout<<"Input "<<i<<": ";
        getfloat(inBuffer[i]);
      }
      std::cout<<"Setting input..."<<std::endl;
      user_nnet->setInput(inBuffer);
      std::cout<<"Inferencing..."<<std::endl;
      user_nnet->infer();
      user_nnet->getOutput(outBuffer);
      std::cout<<"Output: ";
      for (int i = 0; i < user_nnet->getOutputSize(); ++i) {
        std::cout<<outBuffer[i]<<"  ";
      }
      std::cout<<std::endl;
      break;
    case 2:
      //train
      std::cout<<"WARNING: Training directories currently do not work. Please put data in the same directory as the executable. Thanks!"<<std::endl;
      std::cout<<"Input your training data directory: ";
      //getstring(tdir);
      std::cout<<"Input learning rate (alpha): ";
      getfloat(learnrate);
      std::cout<<"Input epochs: ";
      getint(ep);
      user_nnet->trn(learnrate, ep);
      break;
    default:
      std::cout<<"Command not recognized."<<std::endl;
  }
}
