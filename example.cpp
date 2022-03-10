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
#include <utility>

// Define a new Module.
class Net : public torch::nn::Module {
public:
    Net() {
        // Construct and register two Linear submodules.
        fc1 = register_module("fc1", torch::nn::Linear(784, 2));
        sigmoid = register_module("sigmoid", torch::nn::Sigmoid());
    }

    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x) {
        // Use one of many tensor manipulation functions.
        x = fc1->forward(x.reshape({x.size(0), 784}));
        x = sigmoid(x);
        return x;
    }

    // Use one of many "standard library" modules.
    torch::nn::Linear fc1{nullptr};
    torch::nn::Sigmoid sigmoid{nullptr};
};

class myBackward {
private:
    torch::Tensor pred;
    torch::Tensor feature;
    torch::Tensor label;
    torch::Tensor input;
    int learning_rate;

public:
    // Constructor
    myBackward(torch::Tensor pred, torch::Tensor label, torch::Tensor feature, double lr = 0.1)
        : pred(std::move(pred)), label(std::move(label)), feature(std::move(feature)), input(input), learning_rate(lr) {
    }

    myBackward() = default;

    ~myBackward() = default;

    static torch::Tensor calGrad_weight(const torch::Tensor &pred,
                                        const torch::Tensor &label,
                                        const torch::Tensor &input) {
        auto newLabel = label.item<int>();
        torch::Tensor correct = torch::zeros({1, 2}).toType(torch::kLong);
        if (newLabel==1) {
            correct[0][1] = 1;
        } else {
            correct[0][0] = 1;
        }
        auto gradWeight = -1*torch::mm((correct - pred).t(), input.reshape({input.size(0), 784}));

        return gradWeight;
    }

    static torch::Tensor calGrad_bias(const torch::Tensor &pred, const torch::Tensor &label) {
        auto newLabel = label.item<int>();
        torch::Tensor correct = torch::zeros({1, 2});
        if (newLabel==1) {
            correct[0][1] = 1;
        } else {
            correct[0][0] = 1;
        }
        auto gradBias = -1*(correct - pred).squeeze();
        return gradBias;
    }

    static void SGD_UpdateWeight(const torch::Tensor &gradWeight, torch::Tensor &weight, double learning_rate = 0.1) {
        weight = weight - learning_rate*gradWeight;
    }

    static void SGD_UpdateBias(const torch::Tensor &gradBias, torch::Tensor &bias, double learning_rate = 0.1) {
        bias = bias - learning_rate*gradBias;
    }
};

int main() {
    // Create Nets of PartyA and PartyB.
    auto PartyA = std::make_shared<Net>();
    // auto PartyB = std::make_shared<Net>();

    // Create data loader.
    auto data_loader = torch::data::make_data_loader(
        torch::data::datasets::MNIST("../data").map(torch::data::transforms::Stack<>()));

    std::ofstream sfile("./log.txt", ios::out);
    double lr = 0.01;
    torch::Tensor myTarget, predictionA, loss, gradWeight, gradBias;

    for (size_t epoch = 1; epoch <= 10; ++epoch) {
        float lossSum = 0;
        size_t batch_index = 0;
        // Iterate the data loader to yield batches from the dataset.
        for (auto &batch : *data_loader) {
            myTarget = torch::zeros({1}).toType(torch::kLong);
            int labelTarget = batch.target.item<int>();
            if (labelTarget%2==1) {
                myTarget[0] = 1;
            }

            // Execute the model on the input data.
            predictionA = PartyA->forward(batch.data);
            // torch::Tensor predictionB = PartyB->forward(batch.data);

            // partyA的预测值+partyB的预测值
            // torch::Tensor newPrediction = 0.5 * predictionA + 0.5 * predictionB;

            // 计算loss
            loss = torch::cross_entropy_loss(predictionA, myTarget);
            // auto loss = torch::cross_entropy_loss(newPrediction, batch.target);
            lossSum += loss.item<float>();

            // 更新模型
            gradWeight = myBackward::calGrad_weight(predictionA, myTarget, batch.data);
            gradBias = myBackward::calGrad_bias(predictionA, myTarget);
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
            if (batch_index%10000==0) {
                cout << "We have trained " << epoch << " epochs..." << endl;
                sfile << "Epoch: " << epoch << " | Batch: " << batch_index
                      << " | Average Loss: " << lossSum/batch_index << std::endl;
            }
        }
    }

    sfile.close();
    return 0;
}
