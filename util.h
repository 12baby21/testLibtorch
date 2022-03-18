#ifndef _UTIL_H_
#define _UTIL_H_
#include <torch/torch.h>
#include <gmp.h>

// Backward function
class myBackward
{
private:
    double learning_rate;

public:
    myBackward() = default;
    ~myBackward() = default;

    static torch::Tensor calGrad_weight(const torch::Tensor &pred,
                                        const torch::Tensor &diff,
                                        const torch::Tensor &input);

    static torch::Tensor calGrad_bias(const torch::Tensor &pred,
                                      const torch::Tensor &diff);

    static void SGD_UpdateWeight(const torch::Tensor &gradWeight,
                                 torch::Tensor &weight,
                                 double learning_rate);

    static void SGD_UpdateBias(const torch::Tensor &gradBias,
                               torch::Tensor &bias,
                               double learning_rate);
};

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


#endif