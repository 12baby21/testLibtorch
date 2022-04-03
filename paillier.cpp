#include <iostream>
#include <math.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <gmpxx.h>
#include <vector>
#include <thread>

using namespace std;

#define NUM_THREADS 8 // 线程数

extern mpz_class n, g, lambda, mu, nsquare;

mpz_class GenRandomPrime(mp_bitcnt_t bits)
{
    clock_t time = clock();
    mpz_class r;
    gmp_randclass rr(gmp_randinit_default);
    rr.seed(time);
    r = rr.get_z_bits(1024);
    mpz_nextprime(r.get_mpz_t(), r.get_mpz_t());
    return r;
}

void GenKey(mp_bitcnt_t bits, mpz_class &n, mpz_class &g, mpz_class &lambda, mpz_class &mu, mpz_class &nsquare)
{
    // Random State
    clock_t time = clock();
    // gmp_randstate_t grt;
    // gmp_randinit_default(grt);
    // gmp_randseed_ui(grt, time);

    // Generate Two big Primes
    mpz_class PrimeP, PrimeQ;
    gmp_randclass rr(gmp_randinit_default);
    rr.seed(time);
    PrimeP = rr.get_z_bits(1024);
    PrimeQ = rr.get_z_bits(1024);
    mpz_nextprime(PrimeP.get_mpz_t(), PrimeP.get_mpz_t());
    mpz_nextprime(PrimeQ.get_mpz_t(), PrimeQ.get_mpz_t());

    // Generate Public Key (n, g)
    // n = PrimeP * PrimeQ
    // g = n + 1, if PrimeP and PrimeQ have the same length
    n = PrimeP * PrimeQ;
    g = n + 1;

    // Generate Secret Key (lambda, mu)
    // lambda = (PrimeP-1) * (PrimeQ-1)
    // mu = lambda -1
    lambda = (PrimeP - 1) * (PrimeQ - 1);
    mu = lambda - 1;

    // n_2 = n^2
    nsquare = n * n;
}

void Encryption(mpz_class &c, mpz_class &m)
{
    // Random Key
    mpz_class r(2);

    // gm = g^m mod n^2
    // rn = r^n mod n^2
    mpz_class gm;
    mpz_class rn;
    mpz_powm(gm.get_mpz_t(), g.get_mpz_t(), m.get_mpz_t(), nsquare.get_mpz_t());
    mpz_powm(rn.get_mpz_t(), r.get_mpz_t(), n.get_mpz_t(), nsquare.get_mpz_t());

    // c = g^m * r^n mod n^2 = (g^m mod n^2) * (r^n mod n^2) mod n^2
    // c= gm * rn mod n^2
    c = gm * rn;
    c = c % nsquare;
}

void Decryption(mpz_class &res, mpz_class &c)
{
    mpz_class L;
    mpz_powm(L.get_mpz_t(), c.get_mpz_t(), lambda.get_mpz_t(), nsquare.get_mpz_t());
    L = (L - 1) / n;

    mpz_class lambdainvert;
    mpz_invert(lambdainvert.get_mpz_t(), lambda.get_mpz_t(), n.get_mpz_t());
    L = L * lambdainvert;
    L = L % n;
    // L = L * mu
    res = L;
}

// add
void EncryptAdd(mpz_class &res, mpz_class &c1, mpz_class &c2)
{
    res = c1 * c2 % nsquare;
}

// mul  D(c^m) = D(c) * m
void EncryptMul(mpz_class &res, mpz_class &c, mpz_class &m)
{
    mpz_class max_int = n / 3;           // n/3
    mpz_class forNegative = max_int * 2; // 2n/3

    if (cmp(m, forNegative) == 1)
    {
        mpz_class neg_c, neg_scalar;
        mpz_invert(neg_c.get_mpz_t(), c.get_mpz_t(), nsquare.get_mpz_t());
        neg_scalar = n - m;
        mpz_powm(res.get_mpz_t(), neg_c.get_mpz_t(), neg_scalar.get_mpz_t(), nsquare.get_mpz_t());
        return;
    }

    mpz_powm(res.get_mpz_t(), c.get_mpz_t(), m.get_mpz_t(), nsquare.get_mpz_t());
}

void Encode(mpz_class &res, float scalar, const unsigned scale = 1e6)
{
    bool flag = (scalar < 0);
    if (flag)
        scalar = -scalar;
    unsigned after_scale = static_cast<unsigned>(scalar * scale);
    if (flag)
    {
        res = n - after_scale;
    }
    else
    {
        res = after_scale;
    }
}

void Decode(float &res, mpz_class plain, bool isMul, int scale_factor = 1e6)
{
    long ret;
    mpz_class max_int = n / 3;           // n/3
    mpz_class forNegative = max_int * 2; // 2n/3
    int isPositive = cmp(max_int, plain);
    int isNegative = cmp(plain, forNegative);

    if (!isMul)
    {
        if (isNegative == 1)
        {
            mpz_class tmp = plain - n;
            ret = tmp.get_si();
        }
        else if (isPositive == 1)
        {
            ret = plain.get_si();
        }
        else
        {
            cout << "There is a possible OVERFLOW!\n";
        }
    }
    else
    {
        if (isNegative == 1)
            plain = n - plain;
        plain = plain / scale_factor;
        ret = plain.get_si();
        if (isNegative == 1)
            ret = -ret;
    }

    res = static_cast<float>(ret) / scale_factor;
}

void multiEncryptMul(std::vector<mpz_class> &res, mpz_class &c, std::vector<mpz_class> &m, int startIndex, int endIndex)
{
    for (int i = startIndex; i < endIndex; ++i)
    {
        EncryptMul(res[i], c, m[i - startIndex]);
    }
}

void multiDecryption(std::vector<mpz_class> &res, std::vector<mpz_class> &c, int startIndex, int endIndex)
{
    for (int i = startIndex; i < endIndex; ++i)
    {
        Decryption(res[i], c[i]);
    }
}
