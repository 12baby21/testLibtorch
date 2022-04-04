#ifndef _UTIL_H_
#define _UTIL_H_
#include <torch/torch.h>
#include <gmp.h>
#include <gmpxx.h>
#include <vector>

using namespace std;

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
        auto y = x / 100;
        std::cout << x.sizes() << std::endl;
        std::cout << x << std::endl;
        y = sigmoid(y);
        return y;
    }

    // Use one of many "standard library" modules.
    torch::nn::Linear fc1{nullptr};
    torch::nn::Sigmoid sigmoid{nullptr};
};

void myLinear(vector<float> &fc_out, vector<float> &x, vector<float> &weight, vector<float> &bias);
void mySigmoid(vector<float> &out, vector<float> &fc_out);
void mySGDUpdateWeight(vector<float>& weight, vector<float>& grad, float lr);
void mySGDUpdateBias(vector<float>& bias, vector<float>& grad, float lr);
void myCalGradWeight(vector<float> &gradWeight, vector<float> &diff, vector<float> &input);
void myCalGradBias(vector<float> &gradBias, vector<float> diff);
float myCrossEntropyLoss(vector<float> &prediction, char thisLabel);


#endif