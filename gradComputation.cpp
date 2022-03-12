#include <iostream>
#include "util.h"


torch::Tensor myBackward::calGrad_weight(const torch::Tensor &pred,
                                                const int label,
                                                const torch::Tensor &input)
{
    torch::Tensor correct = torch::zeros({1, 2}).toType(torch::kLong);
    if (label == 1)
        correct[0][1] = 1;
    else
        correct[0][0] = 1;
    auto gradWeight = -1 * torch::mm((correct - pred).t(), input.reshape({input.size(0), 784}));

    return gradWeight;
}

torch::Tensor myBackward::calGrad_bias(const torch::Tensor &pred,
                                              const int &label)
{
    torch::Tensor correct = torch::zeros({1, 2});
    if (label == 1)
        correct[0][1] = 1;
    else
        correct[0][0] = 1;
    auto gradBias = -1 * (correct - pred).squeeze();

    return gradBias;
}

void myBackward::SGD_UpdateWeight(const torch::Tensor &gradWeight,
                                         torch::Tensor &weight,
                                         double learning_rate = 0.1)
{
    weight = weight - learning_rate * gradWeight;
}

void myBackward::SGD_UpdateBias(const torch::Tensor &gradBias,
                                       torch::Tensor &bias,
                                       double learning_rate = 0.1)
{
    bias = bias - learning_rate * gradBias;
}