/**
 * Size of weight: [2, 784]
 * Size of bias: [2]
 **/

#include <torch/torch.h>
#include <iostream>
#include <string>
using namespace std;
using namespace torch::autograd;
#include <fstream>
#include <cmath>

// Define a new Module.
class Net : public torch::nn::Module
{
public:
  Net()
  {
    // Construct and register two Linear submodules.
    fc1 = register_module("fc1", torch::nn::Linear(784, 2));
    sigmoid = register_module("sigmoid", torch::nn::Sigmoid());
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x)
  {
    // Use one of many tensor manipulation functions.
    x = fc1->forward(x.reshape({x.size(0), 784}));
    x = sigmoid(x);
    return x;
  }

  // Use one of many "standard library" modules.
  torch::nn::Linear fc1{nullptr};
  torch::nn::Sigmoid sigmoid{nullptr};
};

class myBackward
{
private:
  torch::Tensor pred;
  torch::Tensor feature;
  torch::Tensor label;
  torch::Tensor input;
  int learning_rate;

public:
  // Constructor
  myBackward(torch::Tensor pred, torch::Tensor label, torch::Tensor feature, int lr = 0.1)
      : pred(pred), label(label), feature(feature), input(input), learning_rate(lr)
  {
  }

  myBackward() = default;

  ~myBackward() = default;

  static torch::Tensor calGrad_weight(torch::Tensor pred, torch::Tensor label, torch::Tensor input)
  {
    auto newLabel = label.item<int>();
    torch::Tensor correct = torch::zeros({1, 2});
    if (newLabel == 1)
    {
      correct[0][1] = 1;
    }
    else
    {
      correct[0][0] = 1;
    }

    auto gradWeight = -1 * torch::mm((correct - pred).t(), input.reshape({input.size(0), 784}));

    return gradWeight;
  }

  static torch::Tensor calGrad_bias(torch::Tensor pred, torch::Tensor label)
  {
    auto newLabel = label.item<int>();
    torch::Tensor correct = torch::zeros({1, 2});
    if (newLabel == 1)
    {
      correct[0][1] = 1;
    }
    else
    {
      correct[0][0] = 1;
    }
    auto gradBias = -1 * (correct - pred).squeeze();
    return gradBias;
  }

  static void SGD_UpdateWeight(torch::Tensor &gradWeight, torch::Tensor &weight, int learning_rate = 0.1)
  {
    weight = weight - learning_rate * gradWeight;
  }

  static void SGD_UpdateBias(torch::Tensor gradBias, torch::Tensor &bias, int learning_rate = 0.1)
  {
    bias = bias - learning_rate * gradBias;
  }
};

int main()
{
  // Create Nets of PartyA and PartyB.
  auto PartyA = std::make_shared<Net>();
  // auto PartyB = std::make_shared<Net>();

  // Create data loader.
  auto data_loader = torch::data::make_data_loader(
      torch::data::datasets::MNIST("/home/wjf/Desktop/testLibtorch/data").map(torch::data::transforms::Stack<>()));

  std::ofstream sfile("./log.txt", ios::out);
  int lr = 0.1;
  for (size_t epoch = 1; epoch <= 5; ++epoch)
  {
    float lossSum = 0;
    size_t batch_index = 0;
    // Iterate the data loader to yield batches from the dataset.
    for (auto &batch : *data_loader)
    {
      if (batch.target.item<int>() % 2, 0)
      {
        batch.target.fill_(0);
      }
      else
      {
        batch.target.fill_(1);
      }

      // Execute the model on the input data.
      torch::Tensor predictionA = PartyA->forward(batch.data);
      // torch::Tensor predictionB = PartyB->forward(batch.data);

      // partyA的预测值+partyB的预测值
      // torch::Tensor newPrediction = 0.5 * predictionA + 0.5 * predictionB;

      // 计算loss
      /*
      auto index = (torch::argmax(predictionA)).item<int>();
      float _y = predictionA[0][index].item<float>();
      auto y = (batch.target).item<int>();
      cout << _y << endl;
      cout << y << endl;
      auto loss = y * log2f32(_y) + (1 - y) * log2f32(1 - _y);
      lossSum -= loss;
      */
      auto loss = torch::cross_entropy_loss(predictionA, batch.target);
      // auto loss = torch::cross_entropy_loss(newPrediction, batch.target);
      lossSum += loss.item<float>();

      // 更新模型
      auto gradWeight = myBackward::calGrad_weight(predictionA, batch.target, batch.data);
      auto gradBias = myBackward::calGrad_bias(predictionA, batch.target);
      myBackward::SGD_UpdateWeight(gradWeight, PartyA->fc1->weight, lr);
      myBackward::SGD_UpdateBias(gradBias, PartyA->fc1->bias, lr);
      /*
      auto gradWeight = myBackward::calGrad_weight(newPrediction, batch.target, batch.data);
      auto gradBias = myBackward::calGrad_bias(newPrediction, batch.target);
      myBackward::SGD_UpdateWeight(gradWeight, PartyA->fc1->weight, lr);
      myBackward::SGD_UpdateBias(gradBias, PartyA->fc1->bias, lr);
      myBackward::SGD_UpdateWeight(gradWeight, PartyB->fc1->weight, lr);
      myBackward::SGD_UpdateBias(gradBias, PartyB->fc1->bias, lr);
      */

      ++batch_index;
      // Output the loss and checkpoint every 100 batches.
      if (batch_index % 10000 == 0)
      {
        cout << "We have trained " << epoch << " epochs..." << endl;
        sfile << "Epoch: " << epoch << " | Batch: " << batch_index
              << " | Average Loss: " << lossSum / batch_index << std::endl;
      }
    }
  }

  sfile.close();
  return 0;
}
