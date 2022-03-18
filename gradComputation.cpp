#include <iostream>
#include <gmpxx.h>
#include <vector>
#include "util.h"
#include "paillier.h"

extern mpz_t nsquare;
extern mpz_t n;
extern mpz_t lambda;

torch::Tensor myBackward::calGrad_weight(const torch::Tensor &pred,
                                         const torch::Tensor &diff,
                                         const torch::Tensor &input)
{
    auto gradWeight = -1 * torch::mm(diff.t(), input.reshape({input.size(0), 784}));

    return gradWeight;
}

torch::Tensor myBackward::calGrad_bias(const torch::Tensor &pred,
                                       const torch::Tensor &diff)
{
    auto gradBias = -1 * diff.squeeze();

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
