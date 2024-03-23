#include "defs.cuh"

void ui::run() {
  std::string userinpt;
  while (userinpt != "exit") {
    std::cout<<"This is a test. Type 'exit' to leave."<<std::endl;
    getstring(userinpt);
  }
  //clear this memory before shutdown.
  delete user_nnet;
}
