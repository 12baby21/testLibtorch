#include <torch/torch.h>
#include <iostream>
#include <string>
#include <fstream>
#include <gmpxx.h>
#include "util.h"
#include "paillier.h"
#include <ctime>
#include <chrono>
#include <thread>

using namespace std;
using namespace torch::autograd;
using namespace std::chrono;
#define mnist torch::data::datasets::MNIST

// global variables
mpz_class n, g, lambda, mu, nsquare;
mpz_class R;

int main()
{
    // Set a graph of no grad
    torch::NoGradGuard no_grad;

    // 系统初始化
    GenKey(1024, n, g, lambda, mu, nsquare);
    // 随机掩码R
    // 必须保存在全局，才能计算出来
    R = GenRandomPrime(512);

    // Create Nets of PartyA and PartyB.
    // 自定义初始化模型
    auto PartyA = std::make_shared<Net>();
    auto PartyB = std::make_shared<Net>();

    std::vector<float> weightA(PartyA->fc1->weight.data_ptr<float>(), PartyA->fc1->weight.data_ptr<float>() + PartyA->fc1->weight.numel());
    std::vector<float> biasA(PartyA->fc1->bias.data_ptr<float>(), PartyA->fc1->bias.data_ptr<float>() + PartyA->fc1->bias.numel());
    std::vector<float> weightB(PartyB->fc1->weight.data_ptr<float>(), PartyB->fc1->weight.data_ptr<float>() + PartyB->fc1->weight.numel());
    std::vector<float> biasB(PartyB->fc1->bias.data_ptr<float>(), PartyB->fc1->bias.data_ptr<float>() + PartyB->fc1->bias.numel());
    // Create data loader.
    auto data_loader = torch::data::make_data_loader(
        mnist("../data").map(torch::data::transforms::Stack<>()));
    std::ofstream sfile("../trainLog.txt", ios::out);
    ofstream para("../parameters.txt", ios::out);
    double lr = 0.01;

    mpz_class mask;
    Encryption(mask, R);

    // Options
    at::TensorOptions opts = at::TensorOptions().dtype(torch::kFloat32);

    for (size_t epoch = 1; epoch <= 1; ++epoch)
    {

        float lossSum = 0;
        size_t batch_index = 0;
        // Iterate the data loader to yield batches from the dataset.
        for (auto &batch : *data_loader)
        {
            // 处理输入
            std::vector<float> v_input(batch.data.data_ptr<float>(), batch.data.data_ptr<float>() + batch.data.numel());

            ++batch_index;

            // Modify the label to 0 & 1
            auto myTarget = torch::zeros({1}).toType(torch::kLong);
            int oldLabel = batch.target.item<int>();
            int newLabel = (oldLabel % 2 == 1);
            myTarget[0] = newLabel;

            // Execute the model on the input data.
            vector<float> fc_outA(2);
            vector<float> fc_outB(2);
            vector<float> prediction(2);
            myLinear(fc_outA, v_input, weightA, biasA);
            mySigmoid(fc_outA, fc_outA);
            myLinear(fc_outB, v_input, weightB, biasB);
            mySigmoid(fc_outB, fc_outB);
            for (int i = 0; i < 2; ++i)
            {
                prediction[i] = 0.5 * fc_outA[i] + 0.5 * fc_outB[i];
            }

            torch::Tensor integreated_prediction = torch::from_blob(prediction.data(), {1, 2}, opts).clone();

            // 计算loss
            auto loss = torch::cross_entropy_loss(integreated_prediction, myTarget);
            lossSum += loss.item<float>();

            // 加密(label - newPrediction)并传给B
            vector<long> correct(2);
            if (newLabel == 1)
                correct[1] = 1;
            else
                correct[0] = 1;
            vector<float> diff(2);
            diff[0] = correct[0] - prediction[0];
            diff[1] = correct[1] - prediction[1];
            diff[0] = -1 * diff[0];
            diff[1] = -1 * diff[1];
            cout << "diff[0] = " << diff[0] << endl;
            cout << "diff[1] = " << diff[1] << endl;

            // 计算参与方A的梯度
            vector<float> gradWeightA(784 * 2);
            vector<float> gradBiasA(2);
            myCalGradWeight(gradWeightA, diff, v_input);
            myCalGradBias(gradBiasA, diff);

            // PartyA把diff加密后传给B
            auto start_time = system_clock::now();
            mpz_class enDiff_0, enDiff_1; // 编码后的第0,1维数据
            Encode(enDiff_0, diff[0], 1e6);
            Encode(enDiff_1, diff[1], 1e6);

            std::vector<mpz_class> enDiff = {enDiff_0, enDiff_1};
            for (int i = 0; i < enDiff.size(); ++i)
            {
                Encryption(enDiff[i], enDiff[i]);
            }
            auto end_time = system_clock::now();
            auto duration = duration_cast<microseconds>(end_time - start_time);
            printf("Encrypt diff took %lf s.\n", double(duration.count()) * microseconds::period::num / microseconds::period::den);

            int row = enDiff.size(), column = v_input.size();
            std::vector<mpz_class> mpz_input(column);
            for (int i = 0; i < column; ++i)
            {
                Encode(mpz_input[i], v_input[i], 1e6);
            }

            start_time = system_clock::now();
            // 1.B根据Encrypt(diff)计算加密后weight的梯度，并添加随机掩码R
            std::vector<mpz_class> encrypt_gradWeight_B(row * column);
            thread p1(multiEncryptMul, std::ref(encrypt_gradWeight_B), std::ref(enDiff[0]), std::ref(mpz_input), 0, column);
            thread p2(multiEncryptMul, std::ref(encrypt_gradWeight_B), std::ref(enDiff[1]), std::ref(mpz_input), column, row * column);
            p1.join();
            p2.join();
            for (int i = 0; i < row; ++i)
            {
                for (int j = 0; j < column; ++j)
                {
                    EncryptAdd(encrypt_gradWeight_B[i * column + j], encrypt_gradWeight_B[i * column + j], mask);
                }
            }
            end_time = system_clock::now();
            duration = duration_cast<microseconds>(end_time - start_time);
            printf("B calculate grads of weight using encrypted data took %lf s.\n", double(duration.count()) * microseconds::period::num / microseconds::period::den);

            start_time = system_clock::now();
            // 1.B根据Encrypt(diff)计算加密后bias的梯度，并添加随机掩码R
            std::vector<mpz_class> encrypt_gradBias_B(row);
            for (int i = 0; i < row; ++i)
            {
                encrypt_gradBias_B[i] = enDiff[i];
                EncryptAdd(encrypt_gradBias_B[i], encrypt_gradBias_B[i], mask);
            }
            end_time = system_clock::now();
            duration = duration_cast<microseconds>(end_time - start_time);
            printf("B calculate grads of bias using encrypted data took %lf s.\n", double(duration.count()) * microseconds::period::num / microseconds::period::den);

            start_time = system_clock::now();
            // 2. 把B计算weight的梯度传输给A进行解密，并传回给B
            std::vector<mpz_class> decrypt_gradWeight_B(row * column);
            thread pt1(multiDecryption, std::ref(decrypt_gradWeight_B), std::ref(encrypt_gradWeight_B), 0 * column, 0.25 * column);
            thread pt2(multiDecryption, std::ref(decrypt_gradWeight_B), std::ref(encrypt_gradWeight_B), 0.25 * column, 0.5 * column);
            thread pt3(multiDecryption, std::ref(decrypt_gradWeight_B), std::ref(encrypt_gradWeight_B), 0.5 * column, 0.75 * column);
            thread pt4(multiDecryption, std::ref(decrypt_gradWeight_B), std::ref(encrypt_gradWeight_B), 0.75 * column, 1 * column);
            thread pt5(multiDecryption, std::ref(decrypt_gradWeight_B), std::ref(encrypt_gradWeight_B), 1 * column, 1.25 * column);
            thread pt6(multiDecryption, std::ref(decrypt_gradWeight_B), std::ref(encrypt_gradWeight_B), 1.25 * column, 1.5 * column);
            thread pt7(multiDecryption, std::ref(decrypt_gradWeight_B), std::ref(encrypt_gradWeight_B), 1.5 * column, 1.75 * column);
            thread pt8(multiDecryption, std::ref(decrypt_gradWeight_B), std::ref(encrypt_gradWeight_B), 1.75 * column, 2 * column);
            pt1.join();
            pt2.join();
            pt3.join();
            pt4.join();
            pt5.join();
            pt6.join();
            pt7.join();
            pt8.join();

            // 2. 把B计算bias的梯度传输给A进行解密，并传回给B
            std::vector<mpz_class> decrypt_gradBias_B(row);
            for (int i = 0; i < row; ++i)
            {
                Decryption(decrypt_gradBias_B[i], encrypt_gradBias_B[i]);
            }
            end_time = system_clock::now();
            duration = duration_cast<microseconds>(end_time - start_time);
            printf("A decrypt grads took %lf s.\n", double(duration.count()) * microseconds::period::num / microseconds::period::den);

            // 3. 减去随机掩码并解码 weight
            start_time = system_clock::now();
            std::vector<std::vector<float>> v_gradWeight_B(row, std::vector<float>(column, 0));
            for (int i = 0; i < row; ++i)
            {
                for (int j = 0; j < column; ++j)
                {
                    decrypt_gradWeight_B[i * column + j] = decrypt_gradWeight_B[i * column + j] - R;
                    Decode(v_gradWeight_B[i][j], decrypt_gradWeight_B[i * column + j], true, 1e6);
                }
            }
            std::vector<float> gradWeightB(row * column);
            for (int i = 0; i < row; ++i)
            {
                for (int j = 0; j < column; ++j)
                {
                    gradWeightB[i * column + j] = v_gradWeight_B[i][j];
                }
            }

            // 3. 减去随机掩码并解码 bias
            std::vector<float> v_gradBias_B(row);
            for (int i = 0; i < row; ++i)
            {
                decrypt_gradBias_B[i] = decrypt_gradBias_B[i] - R;
                Decode(v_gradBias_B[i], decrypt_gradBias_B[i], false, 1e6);
            }
            end_time = system_clock::now();
            duration = duration_cast<microseconds>(end_time - start_time);
            printf("B get grads took %lf s.\n", double(duration.count()) * microseconds::period::num / microseconds::period::den);




            // 更新模型A
            mySGDUpdateWeight(weightA, gradWeightA, lr);
            mySGDUpdateBias(biasA, gradBiasA, lr);

            // 更新模型B
            mySGDUpdateWeight(weightB, gradWeightB, lr);
            mySGDUpdateBias(biasB, v_gradBias_B, lr);


            // Output the loss and checkpoint every 100 batches.
            if (batch_index % 2 == 0)
            {
                cout << "We have trained " << epoch << " epochs..." << endl;
                sfile << "Epoch: " << epoch << " | Batch: " << batch_index
                      << " | Average Loss: " << lossSum / batch_index << std::endl;

                // Print parameters

                for (const auto b : biasB)
                {
                    para << b << endl;
                }
            }
        }
    }
    para.close();
    sfile.close();
    return 0;
}
