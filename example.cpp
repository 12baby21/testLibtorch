#include <torch/torch.h>
#include <iostream>
#include <string>
#include <fstream>
#include <gmpxx.h>
#include "util.h"
#include "paillier.h"

using namespace std;
using namespace torch::autograd;

#define mnist torch::data::datasets::MNIST

// global variables
mpz_t n, g, lambda, mu, nsquare;

int main()
{
    // Set a graph of no grad
    torch::NoGradGuard no_grad;

    // Key Initialization
    mpz_init(n);
    mpz_init(g);
    mpz_init(lambda);
    mpz_init(mu);
    mpz_init(nsquare);
    GenKey(1024, n, g, lambda, mu, nsquare);

    // Create Nets of PartyA and PartyB.
    auto PartyA = std::make_shared<Net>();
    auto PartyB = std::make_shared<Net>();

    // partial
    double partial = 0.5;

    // Create data loader.
    auto data_loader = torch::data::make_data_loader(
        mnist("../data").map(torch::data::transforms::Stack<>()));
    auto test_loader = torch::data::make_data_loader(
        mnist("/home/wjf/Desktop/testLibtorch/data", mnist::Mode::kTest).map(torch::data::transforms::Stack<>()));

    std::ofstream sfile("../trainLog.txt", ios::out);
    double lr = 0.005;

    for (size_t epoch = 1; epoch <= 1; ++epoch)
    {
        // 这里需要增加一个随机掩码R
        mpz_t R;
        mpz_init(R);

        float lossSum = 0;
        size_t batch_index = 0;
        // Iterate the data loader to yield batches from the dataset.
        for (auto &batch : *data_loader)
        {
            ++batch_index;

            // Modify the label to 0 & 1
            auto myTarget = torch::zeros({1}).toType(torch::kLong);
            int oldLabel = batch.target.item<int>();
            int newLabel = (oldLabel % 2 == 1);
            myTarget[0] = newLabel;

            // Execute the model on the input data.
            auto predictionA = PartyA->forward(batch.data);
            auto predictionB = PartyB->forward(batch.data);

            // 此处省略了PartyB把预测值传给PartyA

            // partyA的预测值+partyB的预测值
            auto newPrediction = partial * predictionA + (1 - partial) * predictionB;

            // 计算loss
            auto loss = torch::cross_entropy_loss(newPrediction, myTarget);
            lossSum += loss.item<float>();

            // 加密(label - newPrediction)并传给B
            torch::Tensor diff = torch::zeros({1, 2});
            torch::Tensor correct = torch::zeros({1, 2}).toType(torch::kLong);
            if (newLabel == 1)
                correct[0][1] = 1;
            else
                correct[0][0] = 1;
            diff = correct - newPrediction; // sizes = {1, 2}

            // PartyA把diff加密后传给B
            float diff_0 = diff[0][0].item<float>();
            float diff_1 = diff[0][1].item<float>();
            mpz_t enDiff_0; // 编码后的第0维数据
            mpz_init(enDiff_0);
            mpz_t enDiff_1; // 编码后的第1维数据
            mpz_init(enDiff_1);
            Encode(enDiff_0, n, diff_0, 1e6);
            Encode(enDiff_1, n, diff_1, 1e6);
            std::vector<mpz_ptr> enDiff = {enDiff_0, enDiff_1};
            // gmp_printf("diff[0] = %f\ndiff[1] = %f\n", diff_0, diff_1);

            for (int i = 0; i < enDiff.size(); ++i)
            {
                Encryption(enDiff[i], enDiff[i], g, n, nsquare);
                // Decryption(enDiff[i], enDiff[i], lambda, n, nsquare);
                // gmp_printf("enDiff[%d] = %Zd\n", i, enDiff[i]);
                // float tmp;
                // Decode(tmp, n, enDiff[i], false, 1e6);
                // cout << tmp << endl;
            }

            std::vector<float> v_input(batch.data.data_ptr<float>(), batch.data.data_ptr<float>() + batch.data.numel());
            int row = enDiff.size(), column = v_input.size();
            std::vector<mpz_t> mpz_input(column);
            // init mpz_input
            for (int i = 0; i < column; ++i)
            {
                mpz_init(mpz_input[i]);
                Encode(mpz_input[i], n, v_input[i], 1e6);
            }
            /**
             * HERE
             * CORRECT!
             */

            // 计算A的梯度
            auto gradWeight_A = myBackward::calGrad_weight(newPrediction, diff, batch.data);
            auto gradBias_A = myBackward::calGrad_bias(newPrediction, diff);

            // 1. 计算Bweight的梯度(加密后)
            // 计算梯度w时没有乘以-1!
            std::vector<mpz_t> encrypt_gradWeight_B(row * column);
            for (int i = 0; i < row; ++i)
            {
                for (int j = 0; j < column; ++j)
                {
                    mpz_init(encrypt_gradWeight_B[i * column + j]);
                    EncryptMul(encrypt_gradWeight_B[i * column + j], enDiff[i], mpz_input[j], n, nsquare);

                    GenRandom(R, 1024);
                    mpz_t mask; // 加密后的R
                    mpz_init(mask);
                    Encryption(mask, R, g, n, nsquare);
                    EncryptAdd(encrypt_gradWeight_B[i * column + j], encrypt_gradWeight_B[i * column + j], mask, nsquare);
                    // A之后需要解密
                    mpz_clear(mask);
                }
            }

            // 解密解码在B计算weight的梯度
            std::vector<mpz_t> decrypt_gradWeight_B(row * column);
            std::vector<std::vector<float>> v_gradWeight_B(row, std::vector<float>(column, 0));
            for (int i = 0; i < row; ++i)
            {
                for (int j = 0; j < column; ++j)
                {
                    mpz_init(decrypt_gradWeight_B[i * column + j]);
                    Decryption(decrypt_gradWeight_B[i * column + j], encrypt_gradWeight_B[i * column + j], lambda, n, nsquare);
                    Decode(v_gradWeight_B[i][j], n, decrypt_gradWeight_B[i * column + j], true, 1e6);
                    // gmp_printf("decrypt_gradWeight_B[%d][%d] = %Zd\n", i, j, decrypt_gradWeight_B[i * column + j]);
                }
            }

            // 2. 计算Bbias的梯度(加密后)
            // 计算梯度bias时没有乘以-1!
            std::vector<mpz_t> encrypt_gradBias_B(row);
            for (int i = 0; i < row; ++i)
            {
                mpz_init(encrypt_gradBias_B[i]);
                mpz_set(encrypt_gradBias_B[i], enDiff[i]);
                mpz_t mask; // 加密后的R
                mpz_init(mask);
                Encryption(mask, R, g, n, nsquare);
                EncryptAdd(encrypt_gradBias_B[i], encrypt_gradBias_B[i], mask, nsquare);
                // A之后需要解密
                mpz_clear(mask);
            }

            // 解密解码在B计算bias的梯度
            std::vector<mpz_t> decrypt_gradBias_B(row);
            std::vector<float> v_gradBias_B(row);
            for (int i = 0; i < row; ++i)
            {
                mpz_init(decrypt_gradBias_B[i]);
                Decryption(decrypt_gradBias_B[i], encrypt_gradBias_B[i], lambda, n, nsquare);
                Decode(v_gradBias_B[i], n, decrypt_gradBias_B[i], true, 1e6);
                // gmp_printf("decrypt_gradWeight_B[%d][%d] = %Zd\n", i, j, decrypt_gradWeight_B[i * column + j]);
            }

            at::TensorOptions opts = at::TensorOptions().dtype(torch::kFloat32);
            torch::Tensor gradWeight_B = torch::from_blob(v_gradWeight_B.data(), {row, column}, opts).clone();
            torch::Tensor gradBias_B = torch::from_blob(v_gradBias_B.data(), {row}, opts).clone();

            // 更新模型A
            // 1. 解密加了掩码后的梯度值
            // 2. 把解密后的梯度+掩码传给B
            myBackward::SGD_UpdateWeight(gradWeight_A, PartyA->fc1->weight, lr);
            myBackward::SGD_UpdateBias(gradBias_A, PartyA->fc1->bias, lr);

            // 更新模型B
            // 1. B减去掩码
            myBackward::SGD_UpdateWeight(gradWeight_B, PartyB->fc1->weight, lr);
            myBackward::SGD_UpdateBias(gradBias_B, PartyB->fc1->bias, lr);

            // Output the loss and checkpoint every 100 batches.
            if (batch_index % 10000 == 0)
            {
                cout << "We have trained " << epoch << " epochs..." << endl;
                sfile << "Epoch: " << epoch << " | Batch: " << batch_index
                      << " | Average Loss: " << lossSum / batch_index << std::endl;
            }
        }

        // Test after one epoch
        int correct = 0;
        for (auto &testBatch : *test_loader)
        {
            auto myTestTarget = torch::zeros({1}).toType(torch::kLong);
            int testLabel = testBatch.target.item<int>();
            if (testLabel % 2 == 1)
            {
                myTestTarget[0] = 1;
            }

            auto pred = PartyA->forward(testBatch.data);

            // Get the maximum index
            pred.squeeze_();
            torch::Tensor ans = torch::argmax(pred);

            // Calculate accuracy
            bool flag = (ans.item<int>() == myTestTarget.item<int>());
            if (flag)
            {
                correct += 1;
            }
        }
        sfile << "accuracy: " << 1.0 * correct / 10000 << endl;
    }

    // Print parameters
    ofstream para("../parameters.txt", ios::out);
    for (const auto &pair : PartyA->named_parameters())
    {
        para << pair.key() << ": " << pair.value() << endl;
    }
    para.close();

    sfile.close();
    return 0;
}
