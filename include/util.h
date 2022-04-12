#ifndef _UTIL_H_
#define _UTIL_H_
#include <gmp.h>
#include <gmpxx.h>
#include <vector>

using namespace std;

void myLinear(vector<float> &fc_out, vector<float> &x, vector<float> &weight, vector<float> &bias);
void mySigmoid(vector<float> &out, vector<float> &fc_out);
void mySGDUpdateWeight(vector<float>& weight, vector<float>& grad, float lr);
void mySGDUpdateBias(vector<float>& bias, vector<float>& grad, float lr);
void myCalGradWeight(vector<float> &gradWeight, vector<float> &diff, vector<float> &input);
void myCalGradBias(vector<float> &gradBias, vector<float> diff);
float myCrossEntropyLoss(vector<float> &prediction, char thisLabel);


#endif