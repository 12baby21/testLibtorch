#ifndef _PAILLIER_H_
#define _PAILLIER_H_
#include <iostream>
#include <math.h>
#include <algorithm>
#include <stdlib.h>
#include <gmpxx.h>

using namespace std;

mpz_class GenRandomPrime(mp_bitcnt_t bits);
void GenKey(mp_bitcnt_t bits, mpz_class &n, mpz_class &g, mpz_class &lambda, mpz_class &mu, mpz_class &nsquare);
void Encryption(mpz_class& c, mpz_class& m);
void Decryption(mpz_class& res, mpz_class& c);
void EncryptAdd(mpz_class& res, mpz_class& c1, mpz_class& c2);
void EncryptMul(mpz_class& res, mpz_class& c, mpz_class& m);
void Encode(mpz_class& res, float scalar, const unsigned scale = 1e6);
void Decode(float &res, mpz_class plain, bool isMul, int scale_factor = 1e6);
void multiEncryptMul(std::vector<mpz_class>& res, mpz_class& c, std::vector<mpz_class>& m, int startIndex, int endIndex);
void multiDecryption(std::vector<mpz_class>& res, std::vector<mpz_class>& c, int startIndex, int endIndex);
#endif