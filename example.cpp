#include <iostream>
#include <string>
#include <fstream>
#include <gmpxx.h>
#include "util.h"
#include "paillier.h"
#include <ctime>
#include <chrono>
#include <thread>
#include <readMnist.h>
#include <modelCreation.h>
#include "hard_api/include/hard_api.h"

using namespace std;
using namespace torch::autograd;
using namespace std::chrono;

// global variables
mpz_class n, g, lambda, mu, nsquare;
mpz_class R;

int main()
{
    // 硬件初始化
    auto fpga = hard::Hard();

    // 系统初始化
    GenKey(1024, n, g, lambda, mu, nsquare);
    R = GenRandomPrime(512);

    // 写参数
    /* 文件结构
     * code here
     */

    mpz_class mask;
    Encryption(mask, R);
    double lr = 0.01;
    std::ofstream sfile("../trainLog.txt", ios::out);

    // 加载模型
    string addr1 = "../parameters/weightA.dat";
    string addr2 = "../parameters/biasA.dat";
    string addr3 = "../parameters/weightB.dat";
    string addr4 = "../parameters/biasB.dat";
    vector<float> weightA(784 * 2);
    vector<float> biasA(2);
    vector<float> weightB(784 * 2);
    vector<float> biasB(2);
    loadModel<float>(addr1, weightA);
    loadModel<float>(addr2, biasA);
    loadModel<float>(addr3, weightB);
    loadModel<float>(addr4, biasB);

    // 加载并预处理图片和标签
    const string labelDir = "../data/train-labels-idx1-ubyte";
    const string imageDir = "../data/train-images-idx3-ubyte";
    vector<char> labels;
    vector<vector<float>> images;
    readMnistLabel(labelDir, labels);
    readMnistImages(imageDir, images);
    vector<vector<long>> newLabels(labels.size(), vector<long>(2, 0));
    processMnistLabeltoOddandEven(newLabels, labels);
    normalizeMnistImage(images);

    // 开始训练
    for (size_t epoch = 1; epoch <= 1; ++epoch)
    {
        float lossSum = 0;
        // Iterate the data loader to yield batches from the dataset.
        for (int batchIndex = 0; batchIndex < images.size(); ++batchIndex)
        {
            vector<float> img = images[batchIndex];
            // Execute the model on the input data.
            vector<float> fc_outA(2);
            vector<float> fc_outB(2);
            vector<float> prediction(2);
            myLinear(fc_outA, img, weightA, biasA);
            mySigmoid(fc_outA, fc_outA);
            myLinear(fc_outB, img, weightB, biasB);
            mySigmoid(fc_outB, fc_outB);
            for (int i = 0; i < 2; ++i)
            {
                prediction[i] = 0.5 * fc_outA[i] + 0.5 * fc_outB[i];
            }

            // 计算loss
            float loss = myCrossEntropyLoss(prediction, labels[batchIndex]);
            lossSum += loss;

            // 加密(label - newPrediction)并传给B
            vector<float> diff(2);
            diff[0] = newLabels[batchIndex][0] - prediction[0];
            diff[1] = newLabels[batchIndex][1] - prediction[1];
            diff[0] = -1 * diff[0];
            diff[1] = -1 * diff[1];
            cout << "diff[0] = " << diff[0] << endl;
            cout << "diff[1] = " << diff[1] << endl;

            // 计算参与方A的梯度
            vector<float> gradWeightA(784 * 2);
            vector<float> gradBiasA(2);
            myCalGradWeight(gradWeightA, diff, img);
            myCalGradBias(gradBiasA, diff);

            // PartyA把diff加密后传给B
            auto start_time = system_clock::now();
            // 编码后的第0,1维数据
            mpz_class enDiff_0, enDiff_1;
            Encode(enDiff_0, diff[0], 1e6);
            Encode(enDiff_1, diff[1], 1e6);
            std::vector<mpz_class> enDiff = {enDiff_0, enDiff_1};
            for (int i = 0; i < enDiff.size(); ++i)
            {
                // Encryption(enDiff[i], enDiff[i]);
                enDiff[i] = fpga.encrypt(enDiff[i]);
            }
            auto end_time = system_clock::now();
            auto duration = duration_cast<microseconds>(end_time - start_time);
            printf("Encrypt diff took %lf s.\n", double(duration.count()) * microseconds::period::num / microseconds::period::den);

            // 将输入编码
            int row = 2, column = 784;
            std::vector<mpz_class> mpz_input(column);
            for (int i = 0; i < column; ++i)
            {
                Encode(mpz_input[i], img[i], 1e6);
            }

            // [weight * diff + R]
            start_time = system_clock::now();
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

            // [bias * diff + R]
            start_time = system_clock::now();
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
            // 2. decrypt weight and bias
            std::vector<mpz_class> decrypt_gradWeight_B(row * column);
            std::vector<mpz_class> decrypt_gradBias_B(row);
            const size_t poolSize = 8;
            const size_t chunkSize = (row * column) / poolSize;
            vector<thread> threadsTask;
            for (int i = 0; i < poolSize; ++i)
            {
                threadsTask.push_back(thread(multiDecryption, std::ref(decrypt_gradWeight_B), std::ref(encrypt_gradWeight_B), i * chunkSize, (i + 1) * chunkSize));
            }
            for (int i = 0; i < poolSize; ++i)
            {
                threadsTask[i].join();
            }
            for (int i = 0; i < row; ++i)
            {
                // Decryption(decrypt_gradBias_B[i], encrypt_gradBias_B[i]);
                decrypt_gradBias_B[i] = fpga.decrypt(encrypt_gradBias_B[i]);
            }
            end_time = system_clock::now();
            duration = duration_cast<microseconds>(end_time - start_time);
            printf("A decrypt grads took %lf s.\n", double(duration.count()) * microseconds::period::num / microseconds::period::den);

            // 3. weight - R || bias - R
            std::vector<std::vector<float>> v_gradWeight_B(row, std::vector<float>(column, 0));
            std::vector<float> v_gradBias_B(row);
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
            for (int i = 0; i < row; ++i)
            {
                decrypt_gradBias_B[i] = decrypt_gradBias_B[i] - R;
                Decode(v_gradBias_B[i], decrypt_gradBias_B[i], false, 1e6);
            }

            // 更新模型A
            mySGDUpdateWeight(weightA, gradWeightA, lr);
            mySGDUpdateBias(biasA, gradBiasA, lr);

            // 更新模型B
            mySGDUpdateWeight(weightB, gradWeightB, lr);
            mySGDUpdateBias(biasB, v_gradBias_B, lr);

            // Output the loss and checkpoint every 100 batches.
            if ((batchIndex + 1) % 2 == 0)
            {
                sfile << "Epoch: " << epoch << " | Batch: " << batchIndex + 1
                      << " | Average Loss: " << lossSum / (batchIndex + 1) << std::endl;
            }
        }
    }
    sfile.close();
    return 0;
}
