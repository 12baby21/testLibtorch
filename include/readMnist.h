#ifndef _READMNIST_H_
#define _READMNIST_H_
#include <vector>
#include <string>

using namespace std;

int reverseEndian(int i);
bool readMnistLabel(const string filename, vector<char>& labels);
bool readMnistImages(const string filename, vector<vector<float>>& images);
void processMnistLabeltoOddandEven(vector<vector<long>>& newLabels, vector<char>& labels);
void normalizeMnistImage(vector<vector<float>>& images);

#endif