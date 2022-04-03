#include <iostream>
#include <gmpxx.h>
#include <vector>
#include "util.h"
#include "paillier.h"
#include <cmath>
extern mpz_t nsquare;
extern mpz_t n;
extern mpz_t lambda;

void myLinear(vector<float> &fc_out, vector<float> &x, vector<float> &weight, vector<float> &bias)
{
    int m = fc_out.size();
    int n = x.size();
    for (int i = 0; i < m; ++i)
    {
        float tmp = 0;
        for (int j = 0; j < n; ++j)
        {
            tmp += weight[i * n + j] * x[j];
        }
        tmp += bias[i];
        fc_out[i] = tmp;
    }
}

void mySigmoid(vector<float> &out, vector<float> &fc_out)
{
    for (int i = 0; i < out.size(); ++i)
        out[i] = 1 / (1 + exp(-fc_out[i]));
}

void mySGDUpdateWeight(vector<float> &weight, vector<float> &grad, float lr)
{
    for (int i = 0; i < weight.size(); ++i)
    {
        weight[i] = weight[i] - lr * grad[i];
    }
}

void mySGDUpdateBias(vector<float> &bias, vector<float> &grad, float lr)
{
    for (int i = 0; i < bias.size(); ++i)
    {
        bias[i] = bias[i] - lr * grad[i];
    }
}

void myCalGradWeight(vector<float> &gradWeight, vector<float> &diff, vector<float> &input)
{
    int m = diff.size();
    int n = input.size();

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            gradWeight[i * n + j] = diff[i] * input[j];
        }
    }
}

void myCalGradBias(vector<float> &gradBias, vector<float> diff)
{
    int n = diff.size();
    for(int i = 0; i < n; ++ i)
    {
        gradBias[i] = diff[i];
    }
}