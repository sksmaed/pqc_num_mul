#include <stdio.h>
#include <gmp.h>

void gmp_mul(size_t n, const void *in1, const void *in2, void *out) {
    mpz_t a, b, result;
    mpz_init(a);
    mpz_init(b);
    mpz_init(result);

    mpz_import(a, n / 8, 1, 1, 0, 0, in1);
    mpz_import(b, n / 8, 1, 1, 0, 0, in2);
    mpz_mul(result, a, b);
    mpz_export(out, NULL, 1, 1, 0, 0, result);

    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(result);
}
