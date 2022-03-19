#include <iostream>
#include <math.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <gmpxx.h>
#include <vector>

using namespace std;

void GenKey(mp_bitcnt_t bits, mpz_ptr n, mpz_ptr g, mpz_ptr lambda, mpz_ptr mu, mpz_ptr n_2)
{
    // Random State
    clock_t time = clock();
    gmp_randstate_t grt;
    gmp_randinit_default(grt);
    gmp_randseed_ui(grt, time);

    // Generate Two big Primes
    mpz_t PrimeP, PrimeQ;
    mpz_init(PrimeP);
    mpz_init(PrimeQ);
    mpz_urandomb(PrimeP, grt, bits);
    mpz_urandomb(PrimeQ, grt, bits);
    mpz_setbit(PrimeP, bits);
    mpz_setbit(PrimeQ, bits);
    mpz_nextprime(PrimeP, PrimeP);
    mpz_nextprime(PrimeQ, PrimeQ);

    // Generate Public Key (n, g)
    // n = PrimeP * PrimeQ
    // g = n + 1, if PrimeP and PrimeQ have the same length
    mpz_mul(n, PrimeP, PrimeQ);
    mpz_add_ui(g, n, 1);

    // PrimeP = PrimeP-1
    // PrimeQ = PrimeQ-1
    mpz_sub_ui(PrimeP, PrimeP, 1);
    mpz_sub_ui(PrimeQ, PrimeQ, 1);

    // Generate Secret Key (lambda, mu)
    // lambda = (PrimeP-1) * (PrimeQ-1)
    // mu = lambda -1
    mpz_mul(lambda, PrimeP, PrimeQ);
    mpz_sub_ui(mu, lambda, 1);

    // n_2 = n^2
    mpz_mul(n_2, n, n);
}

void Encryption(mpz_ptr c, mpz_ptr m, mpz_ptr g, mpz_ptr n, mpz_ptr n_2)
{
    // Random Key
    mpz_t r;
    mpz_init_set_ui(r, 2);

    // gm = g^m mod n^2
    // rn = r^n mod n^2
    mpz_t gm;
    mpz_t rn;
    mpz_init(gm);
    mpz_init(rn);
    mpz_powm(gm, g, m, n_2);
    mpz_powm(rn, r, n, n_2);
    // c = g^m * r^n mod n^2 = (g^m mod n^2) * (r^n mod n^2) mod n^2
    // c= gm * rn mod n^2
    mpz_mul(c, gm, rn);
    mpz_mod(c, c, n_2);
}

void Decryption(mpz_ptr res, mpz_ptr c, mpz_ptr lambda, mpz_ptr n, mpz_ptr n_2)
{
    mpz_t l;
    mpz_init(l);
    mpz_powm(l, c, lambda, n_2);
    mpz_sub_ui(l, l, 1);
    mpz_div(l, l, n);

    mpz_t lambdainvert;
    mpz_init(lambdainvert);
    mpz_invert(lambdainvert, lambda, n);
    // mpz_mod(lambdainvert, lambdainvert, n);
    mpz_mul(l, l, lambdainvert);
    mpz_mod(res, l, n);
}

// add
void EncryptAdd(mpz_ptr res, mpz_ptr c1, mpz_ptr c2, mpz_ptr nsquare)
{
    mpz_mul(res, c1, c2);
    mpz_mod(res, res, nsquare);
}

// mul  D(c^m) = D(c) * m
void EncryptMul(mpz_ptr res, mpz_ptr c, mpz_ptr m, mpz_ptr n, mpz_ptr nsquare)
{
    // gmp_printf("c = %Zd\nm = %Zd\n", c, m);
    mpz_t max_int;     // n/3
    mpz_t forNegative; // 2n/3
    mpz_init(max_int);
    mpz_init(forNegative);

    mpz_div_ui(max_int, n, 3);
    mpz_mul_ui(forNegative, max_int, 2);

    if (mpz_cmp(m, forNegative) == 1)
    {
        mpz_t neg_c, neg_scalar;
        mpz_init(neg_c);
        mpz_init(neg_scalar);
        mpz_invert(neg_c, c, nsquare);
        mpz_sub(neg_scalar, n, m);
        mpz_powm(res, neg_c, neg_scalar, nsquare);
        return;
    }

    mpz_powm(res, c, m, nsquare);
}

void Encode(mpz_ptr res, mpz_ptr n, float scalar, int scale = 1e6)
{
    bool flag = (scalar < 0);
    if (flag)
        scalar = -scalar;
    unsigned after_scale = static_cast<unsigned>(scalar * scale);
    mpz_t tmp1;
    mpz_init(tmp1);
    if (flag)
    {
        mpz_sub_ui(res, n, after_scale);
    }
    else
    {
        mpz_set_ui(res, after_scale);
    }
}

void Decode(float &res, mpz_ptr n, mpz_ptr plain, bool isMul, int scale_factor = 1e6)
{
    int ret;
    mpz_t max_int;     // n/3
    mpz_t forNegative; // 2n/3
    mpz_init(max_int);
    mpz_init(forNegative);
    mpz_div_ui(max_int, n, 3);
    mpz_mul_ui(forNegative, max_int, 2);
    int isPositive = mpz_cmp(max_int, plain);
    int isNegative = mpz_cmp(plain, forNegative);

    if (!isMul)
    {
        if (isNegative == 1)
        {
            mpz_t tmp;
            mpz_init(tmp);
            mpz_sub(tmp, plain, n);
            ret = mpz_get_si(tmp);
            mpz_clear(tmp);
        }
        else if (isPositive == 1)
        {
            ret = mpz_get_si(plain);
        }
        else
        {
            cout << "There is a possible OVERFLOW!\n";
        }
    }
    else
    {
        if (isNegative == 1)
            mpz_sub(plain, n, plain);
        mpz_div_ui(plain, plain, scale_factor);
        ret = mpz_get_si(plain);
        if (isNegative == 1)
            ret = -ret;
    }

    res = static_cast<float>(ret) / scale_factor;
    mpz_clear(max_int);     // n/3
    mpz_clear(forNegative); // 2n/3
}

// 生成随机掩码
void GenRandom(mpz_ptr res, int bits)
{
    // Random State
    clock_t time = clock();
    gmp_randstate_t grt;
    gmp_randinit_default(grt);
    gmp_randseed_ui(grt, time);

    // Generate a random number R
    mpz_urandomb(res, grt, bits);
    mpz_setbit(res, bits);
    mpz_nextprime(res, res);
}