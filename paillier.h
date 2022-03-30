#ifndef _PAILLIER_H_
#define _PAILLIER_H_
#include <iostream>
#include <math.h>
#include <algorithm>
#include <stdlib.h>
#include <gmpxx.h>

using namespace std;

void GenKey(mp_bitcnt_t bits, mpz_ptr n, mpz_ptr g, mpz_ptr lambda, mpz_ptr mu, mpz_ptr n_2);
void Encryption(mpz_ptr c, mpz_ptr m, mpz_ptr g, mpz_ptr n, mpz_ptr n_2);
void Decryption(mpz_ptr res, mpz_ptr c, mpz_ptr lambda, mpz_ptr n, mpz_ptr n_2);
void EncryptAdd(mpz_ptr res, mpz_ptr c1, mpz_ptr c2, mpz_ptr n_2);
void EncryptMul(mpz_ptr res, mpz_ptr c, mpz_ptr m, mpz_ptr n, mpz_ptr nsquare);
void Encode(mpz_ptr res, mpz_ptr n, float scalar, int scale = 1e6);
void Decode(float &res, mpz_ptr n, mpz_ptr plain, bool isMul, int scale_factor = 1e6);
void GenRandom(mpz_ptr res, int bits);
void multiEncryptMul(std::vector<mpz_t>& res, const mpz_ptr c, std::vector<mpz_t>& m, const mpz_ptr n, const mpz_ptr nsquare, int start);
void multiDecryption(std::vector<mpz_t>& res, std::vector<mpz_t>& c, mpz_ptr lambda, mpz_ptr n, mpz_ptr nsquare, int start);
#endif