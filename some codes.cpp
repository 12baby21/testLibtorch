  auto test_loader = torch::data::make_data_loader(
      torch::data::datasets::MNIST("/home/wjf/Desktop/testLibtorch/data", torch::data::datasets::MNIST::Mode::kTest).map(torch::data::transforms::Stack<>()));



int correct = 0;
for (auto &testBatch : *test_loader)
{
    if (torch::equal(testBatch.target % 2, tmp))
    {
        testBatch.target.fill_(0);
    }
    else
    {
        testBatch.target.fill_(1);
    }

    auto pred = PartyA->forward(testBatch.data);

    pred.squeeze_();
    torch::Tensor ans = torch::argmax(pred);

    bool flag = torch::equal(ans, testBatch.target);

    if (flag)
    {
        correct += 1;
    }
}
cout << correct << endl;
sfile << "accuracy: " << (static_cast<double>(correct) / 10000.0) << endl;