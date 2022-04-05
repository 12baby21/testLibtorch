#ifndef _MODELCREATION_H_
#define _MODELCREATION_H_

#include <iostream>
#include <vector>
#include <fstream>
#include <time.h>
#include <string>

using namespace std;

template<typename T>
void writeModel(string fileAddr, vector<T>& parameters)
{
    ofstream modelFileOut;
    cout << fileAddr << endl;
    modelFileOut.open(fileAddr.c_str(), ios::out | ios::binary);
    int n = parameters.size();
    for(int i = 0; i < n; ++i)
    {
        modelFileOut.write(reinterpret_cast<char*>(&parameters[i]), sizeof(T));
    }
    modelFileOut.close();
}


template<typename T>
void loadModel(string fileAddr, vector<T>& parameters)
{
    ifstream modelFileIn;
    modelFileIn.open(fileAddr.c_str(), ios::out | ios::binary);
    int n = parameters.size();
    for(int i = 0; i < n; ++i)
    {
        modelFileIn.read(reinterpret_cast<char*>(&parameters[i]), sizeof(T));
    }
    modelFileIn.close();
}

template<typename T>
void createModel(vector<T>& parameters)
{
    srand((int)time(0));
    int n = parameters.size();
    for(int i = 0; i < n; ++i)
    {
        parameters[i] = rand() % 3;
    }
}

template<typename T>
void createModel(vector<float>& parameters)
{
    srand((int)time(0));
    int n = parameters.size();
    for(int i = 0; i < n; ++i)
    {
        parameters[i] = (rand() % 10) / 10.0 - 0.5;
    }
}


#endif