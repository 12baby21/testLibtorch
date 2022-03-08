#include <torch/torch.h>
#include <iostream>
#include <string>
using namespace std;
using namespace torch::autograd;

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
    cout << x.sizes() << endl;
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
  myBackward(torch::Tensor pred, torch::Tensor label, torch::Tensor feature, int lr = 0.1) // diff = y - ^y
      : pred(pred), label(label), feature(feature), input(input), learning_rate(lr)
  {
  }

  myBackward() = default;

  ~myBackward() = default;

  static torch::Tensor calGrad_weight(torch::Tensor pred, torch::Tensor label, torch::Tensor input)
  {
    input.reshape({input.size(0), 784});
    auto gradWeight = (label - pred[0][1]) * input;
    return gradWeight;
  }

  static torch::Tensor calGrad_bias(torch::Tensor pred, torch::Tensor label)
  {
    auto gradBias = (label - pred[0][1]);
    return gradBias;
  }

  static torch::Tensor SGD_UpdateWeight(torch::Tensor gradWeight, torch::Tensor weight, int learning_rate = 0.1)
  {
    weight = weight + 0.1 * gradWeight;
  }

  static torch::Tensor SGD_UpdateBias(torch::Tensor gradBias, torch::Tensor bias, int learning_rate = 0.1)
  {
    1 + 2;
  }
};

int main()
{
  // Create Nets of PartyA and PartyB.
  auto PartyA = std::make_shared<Net>();
  auto PartyB = std::make_shared<Net>();

  // Create data loader.
  auto data_loader = torch::data::make_data_loader(
      torch::data::datasets::MNIST("/home/wjf/Desktop/testLibtorch/data").map(torch::data::transforms::Stack<>()));

  // Instantiate an SGD optimization algorithm to update our Net's parameters.
  torch::optim::SGD optimA(PartyA->parameters(), /*lr=*/0.01);
  torch::optim::SGD optimB(PartyB->parameters(), /*lr=*/0.01);

  for (size_t epoch = 1; epoch <= 5; ++epoch)
  {
    size_t batch_index = 0;
    // Iterate the data loader to yield batches from the dataset.
    for (auto &batch : *data_loader)
    {
      torch::Tensor tmp = torch::zeros_like(batch.target);
      if (torch::equal(batch.target % 2, tmp))
      {
        batch.target.fill_(0);
      }
      else
      {
        batch.target.fill_(1);
      }

      // Reset gradients.
      optimA.zero_grad();
      optimB.zero_grad();
      // Execute the model on the input data.
      torch::Tensor predictionA = PartyA->forward(batch.data);
      torch::Tensor predictionB = PartyB->forward(batch.data);

      // partyA的预测值+partyB的预测值
      torch::Tensor newPrediction = 0.5 * predictionA + 0.5 * predictionB;

      auto loss = torch::cross_entropy_loss(newPrediction, batch.target);

      loss.backward();

      // Update the parameters based on the calculated gradients.
      optimA.step();
      optimB.step();

      // Output the loss and checkpoint every 100 batches.
      if (++batch_index % 100 == 0)
      {
        std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                  << " | Loss: " << loss.item<float>() << std::endl;
        // Serialize your model periodically as a checkpoint.
        torch::save(PartyA, "netA.pt");
        torch::save(PartyB, "netB.pt");
      }
    }
  }
  return 0;
}