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
    fc1 = register_module("fc1", torch::nn::Linear(784, 10));
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x)
  {
    // Use one of many tensor manipulation functions.
    x = fc1->forward(x.reshape({x.size(0), 784}));
    return x;
  }

  // Use one of many "standard library" modules.
  torch::nn::Linear fc1{nullptr};
};



// Inherit from Function
class myLinear : public Function<myLinear> {
 public:
  // bias is an optional argument
  static torch::Tensor forward(
      AutogradContext *ctx, torch::Tensor input, torch::Tensor weight, torch::Tensor bias = torch::Tensor()) {
    ctx->save_for_backward({input, weight, bias});
    auto output = input.mm(weight.t());
    if (bias.defined()) {
      output += bias.unsqueeze(0).expand_as(output);
    }
    return output;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];
    auto bias = saved[2];

    auto grad_output = grad_outputs[0];
    auto grad_input = grad_output.mm(weight);
    auto grad_weight = grad_output.t().mm(input);
    auto grad_bias = torch::Tensor();
    if (bias.defined()) {
      grad_bias = grad_output.sum(0);
    }

    return {grad_input, grad_weight, grad_bias};
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
      // Reset gradients.
      optimA.zero_grad();
      optimB.zero_grad();
      // Execute the model on the input data.
      torch::Tensor predictionA = PartyA->forward(batch.data);
      torch::Tensor predictionB = PartyB->forward(batch.data);

      // partyA的预测值+partyB的预测值
      torch::Tensor newPrediction = 0.5 * predictionA + 0.5 * predictionB;
      newPrediction.detach();

      // Compute a loss value to judge the prediction of our model.
/**
      torch::Tensor lossA = torch::cross_entropy_loss(predictionA, batch.target);
      torch::Tensor lossB = torch::cross_entropy_loss(predictionB, batch.target);
**/
      torch::Tensor loss = torch::cross_entropy_loss(newPrediction, batch.target);

      // Compute gradients of the loss w.r.t. the parameters of our model.
/**
      lossA.backward();
      lossB.backward();
**/
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