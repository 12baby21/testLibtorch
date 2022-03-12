#include <torch/torch.h>
#include <iostream>
#include <string>
#include <fstream>
#include "util.h"

#define mnist torch::data::datasets::MNIST

using namespace std;
using namespace torch::autograd;

int main()
{
    // Set a graph of no grad
    torch::NoGradGuard no_grad;

    // Create Nets of PartyA and PartyB.
    auto PartyA = std::make_shared<Net>();
    auto PartyB = std::make_shared<Net>();

    // Create data loader.
    auto data_loader = torch::data::make_data_loader(
        mnist("../data").map(torch::data::transforms::Stack<>()));
    auto test_loader = torch::data::make_data_loader(
        mnist("/home/wjf/Desktop/testLibtorch/data", mnist::Mode::kTest).map(torch::data::transforms::Stack<>()));

    std::ofstream sfile("../trainLog.txt", ios::out);
    double lr = 0.01;
    torch::Tensor myTarget, predictionA, loss, gradWeight, gradBias;

    for (size_t epoch = 1; epoch <= 10; ++epoch)
    {
        float lossSum = 0;
        size_t batch_index = 0;
        // Iterate the data loader to yield batches from the dataset.
        for (auto &batch : *data_loader)
        {
            myTarget = torch::zeros({1}).toType(torch::kLong);

            int oldLabel = batch.target.item<int>();
            int newLabel = (oldLabel % 2 == 1);
            myTarget[0] = newLabel;

            // Execute the model on the input data.
            predictionA = PartyA->forward(batch.data).detach();
            // torch::Tensor predictionB = PartyB->forward(batch.data);

            // partyA的预测值+partyB的预测值
            // torch::Tensor newPrediction = 0.5 * predictionA + 0.5 * predictionB;

            // 计算loss
            loss = torch::cross_entropy_loss(predictionA, myTarget).detach();
            // auto loss = torch::cross_entropy_loss(newPrediction, batch.target);
            lossSum += loss.item<float>();

            // 更新模型
            gradWeight = myBackward::calGrad_weight(predictionA, newLabel, batch.data).detach();
            gradBias = myBackward::calGrad_bias(predictionA, newLabel).detach();
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
        int correct = 0;
        for (auto &testBatch : *test_loader)
        {
            auto myTestTarget = torch::zeros({1}).toType(torch::kLong);
            int testLabelTarget = testBatch.target.item<int>();
            if (testLabelTarget % 2 == 1)
            {
                myTestTarget[0] = 1;
            }

            auto pred = PartyA->forward(testBatch.data);
            pred.squeeze_();

            torch::Tensor ans = torch::argmax(pred);
            bool flag = (ans.item<int>() == myTestTarget.item<int>());

            if (flag)
            {
                correct += 1;
            }
        }
        sfile << "accuracy: " << correct << endl;
    }

    sfile.close();
    return 0;
}
