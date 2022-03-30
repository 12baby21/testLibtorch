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

    std::vector<float> weightA(PartyA->fc1->weight.data_ptr<float>(), PartyA->fc1->weight.data_ptr<float>() + PartyA->fc1->weight.numel());
    std::vector<float> biasA(PartyA->fc1->bias.data_ptr<float>(), PartyA->fc1->bias.data_ptr<float>() + PartyA->fc1->bias.numel());
    std::vector<float> weightB(PartyB->fc1->weight.data_ptr<float>(), PartyB->fc1->weight.data_ptr<float>() + PartyB->fc1->weight.numel());
    std::vector<float> biasB(PartyB->fc1->bias.data_ptr<float>(), PartyB->fc1->bias.data_ptr<float>() + PartyB->fc1->bias.numel());
    // Create data loader.
    auto data_loader = torch::data::make_data_loader(
        mnist("../data").map(torch::data::transforms::Stack<>()));
    std::ofstream sfile("../trainLog.txt", ios::out);
    double lr = 0.005;

    // Options
    at::TensorOptions opts = at::TensorOptions().dtype(torch::kFloat32);

    for (size_t epoch = 1; epoch <= 1; ++epoch)
    {
        // 随机掩码R
        // 必须保存在全局，才能计算出来
        mpz_t R;
        mpz_init(R);
        GenRandom(R, 512);

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
            for(int i = 0; i < 2; ++i) {
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
            diff[0] = -diff[0];
            diff[1] = -diff[1];

            // 计算参与方A的梯度
            vector<float> gradWeightA(784*2);
            vector<float> gradBiasA(2);
            myCalGradWeight(gradWeightA, diff, v_input);
            myCalGradBias(gradBiasA, diff);


            // PartyA把diff加密后传给B
            auto start_time = system_clock::now();
            mpz_t enDiff_0; // 编码后的第0维数据
            mpz_init(enDiff_0);
            mpz_t enDiff_1; // 编码后的第1维数据
            mpz_init(enDiff_1);
            Encode(enDiff_0, n, diff[0], 1e6);
            Encode(enDiff_1, n, diff[1], 1e6);
            std::vector<mpz_ptr> enDiff = {enDiff_0, enDiff_1};

            for (int i = 0; i < enDiff.size(); ++i)
            {
                Encryption(enDiff[i], enDiff[i], g, n, nsquare);
            }
            auto end_time = system_clock::now();
            auto duration = duration_cast<microseconds>(end_time - start_time);
            printf("Encrypt diff took %lf s.\n", double(duration.count()) * microseconds::period::num / microseconds::period::den);

            int row = enDiff.size(), column = v_input.size();
            std::vector<mpz_t> mpz_input(column);
            for (int i = 0; i < column; ++i)
            {
                mpz_init(mpz_input[i]);
                Encode(mpz_input[i], n, v_input[i], 1e6);
            }

            start_time = system_clock::now();
            // 1.B根据Encrypt(diff)计算加密后weight的梯度，并添加随机掩码R
            // mask = Encrypt(R)
            // EncryptMul改成多线程？
            std::vector<mpz_t> encrypt_gradWeight_B(row * column);
            // for (int i = 0; i < row; ++i)
            // {
            //     multiEncryptMul(encrypt_gradWeight_B, enDiff[i], mpz_input, n, nsquare, i * column);
            // }
            thread p1(multiEncryptMul, std::ref(encrypt_gradWeight_B), enDiff[0], std::ref(mpz_input), n, nsquare, 0);
            thread p2(multiEncryptMul, std::ref(encrypt_gradWeight_B), enDiff[1], std::ref(mpz_input), n, nsquare, column);
            p1.join();
            p2.join();
            for (int i = 0; i < row; ++i)
            {
                mpz_t mask; // 加密后的R
                mpz_init(mask);
                Encryption(mask, R, g, n, nsquare);
                for (int j = 0; j < column; ++j)
                {

                    for (int j = 0; j < column; ++j)
                    {
                        EncryptAdd(encrypt_gradWeight_B[i * column + j], encrypt_gradWeight_B[i * column + j], mask, nsquare);
                    }
                }
                mpz_clear(mask);
            }
            end_time = system_clock::now();
            duration = duration_cast<microseconds>(end_time - start_time);
            printf("B calculate grads of weight using encrypted data took %lf s.\n", double(duration.count()) * microseconds::period::num / microseconds::period::den);
            start_time = system_clock::now();

            // 1.B根据Encrypt(diff)计算加密后bias的梯度，并添加随机掩码R
            // mask = Encrypt(R)
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
            end_time = system_clock::now();
            duration = duration_cast<microseconds>(end_time - start_time);
            printf("B calculate grads of bias using encrypted data took %lf s.\n", double(duration.count()) * microseconds::period::num / microseconds::period::den);

            start_time = system_clock::now();
            // 2. 把B计算weight的梯度传输给A进行解密，并传回给B
            std::vector<mpz_t> decrypt_gradWeight_B(row * column);
            thread pt1(multiDecryption, std::ref(decrypt_gradWeight_B), std::ref(encrypt_gradWeight_B), lambda, n, nsquare, 0 * column);
            thread pt2(multiDecryption, std::ref(decrypt_gradWeight_B), std::ref(encrypt_gradWeight_B), lambda, n, nsquare, 0.25 * column);
            thread pt3(multiDecryption, std::ref(decrypt_gradWeight_B), std::ref(encrypt_gradWeight_B), lambda, n, nsquare, 0.5 * column);
            thread pt4(multiDecryption, std::ref(decrypt_gradWeight_B), std::ref(encrypt_gradWeight_B), lambda, n, nsquare, 0.75 * column);
            thread pt5(multiDecryption, std::ref(decrypt_gradWeight_B), std::ref(encrypt_gradWeight_B), lambda, n, nsquare, 1 * column);
            thread pt6(multiDecryption, std::ref(decrypt_gradWeight_B), std::ref(encrypt_gradWeight_B), lambda, n, nsquare, 1.25 * column);
            thread pt7(multiDecryption, std::ref(decrypt_gradWeight_B), std::ref(encrypt_gradWeight_B), lambda, n, nsquare, 1.5 * column);
            thread pt8(multiDecryption, std::ref(decrypt_gradWeight_B), std::ref(encrypt_gradWeight_B), lambda, n, nsquare, 1.75 * column);
            pt1.join();
            pt2.join();
            pt3.join();
            pt4.join();
            pt5.join();
            pt6.join();
            pt7.join();
            pt8.join();

            // 2. 把B计算bias的梯度传输给A进行解密，并传回给B
            std::vector<mpz_t> decrypt_gradBias_B(row);
            for (int i = 0; i < row; ++i)
            {
                mpz_init(decrypt_gradBias_B[i]);
                Decryption(decrypt_gradBias_B[i], encrypt_gradBias_B[i], lambda, n, nsquare);
                // gmp_printf("decrypt_gradWeight_B[%d][%d] = %Zd\n", i, j, decrypt_gradWeight_B[i * column + j]);
            }
            end_time = system_clock::now();
            duration = duration_cast<microseconds>(end_time - start_time);
            printf("A decrypt grads took %lf s.\n", double(duration.count()) * microseconds::period::num / microseconds::period::den);

            start_time = system_clock::now();
            // 3. 减去随机掩码并解码 weight
            std::vector<std::vector<float>> v_gradWeight_B(row, std::vector<float>(column, 0));
            for (int i = 0; i < row; ++i)
            {
                for (int j = 0; j < column; ++j)
                {
                    mpz_sub(decrypt_gradWeight_B[i * column + j], decrypt_gradWeight_B[i * column + j], R);
                    Decode(v_gradWeight_B[i][j], n, decrypt_gradWeight_B[i * column + j], true, 1e6);
                    // printf("v_gradWeight_B[%d][%d] = %f ", i, j, v_gradWeight_B[i][j]);
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
                mpz_sub(decrypt_gradBias_B[i], decrypt_gradBias_B[i], R);
                Decode(v_gradBias_B[i], n, decrypt_gradBias_B[i], true, 1e6);
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
            if (batch_index % 1 == 0)
            {
                cout << "We have trained " << epoch << " epochs..." << endl;
                sfile << "Epoch: " << epoch << " | Batch: " << batch_index
                      << " | Average Loss: " << lossSum / batch_index << std::endl;

                // Print parameters
                ofstream para("../parameters.txt", ios::out);
                for (const auto &pair : PartyB->named_parameters())
                {
                    para << pair.key() << ": " << pair.value() << endl;
                }
                para.close();
            }
        }

        mpz_clear(R);
    }

    sfile.close();
    return 0;
}
