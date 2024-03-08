#include "defs.cuh"

neuralnet::neuralnet(int ins, int ops, int hls, int hlc, std::string n) {
  name = n;
  inputSize = ins;
  outputSize = ops;
  hiddenLayerSize = hls;
  hiddenLayerCount = hlc;
  input = new float[inputSize];
  output = new float[outputSize];

  //I'm sure there's a more elegant way to do this, but I've had no luck with it.
  hiddenlayers = new layer[hiddenLayerCount];
  for (int i = 0; i < hiddenLayerCount; ++i) {
    hiddenlayers[i].setInputSize(hiddenLayerSize);
    hiddenlayers[i].setOutputSize(hiddenLayerSize);
  }
  hiddenlayers[0].setInputSize(inputSize);
  hiddenlayers[hiddenLayerCount-1].setOutputSize(outputSize);
  for (int i = 0; i < hiddenLayerCount; ++i) {
    hiddenlayers[i].allocLayer();
  }
}
