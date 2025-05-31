#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <string.h>
#include <assert.h>
#include <inttypes.h>
#include "../headers/mul.h"
#include "../headers/constant.h"
#include "../headers/utils.h"

void gmp_mul(size_t n_bytes, const void *in1, const void *in2, void *out) {
    mpz_t a, b, result;
    mpz_init(a);
    mpz_init(b);
    mpz_init(result);

    // 直接以 uint16_t 作為 limb 單位
    size_t n_limbs = n_bytes / sizeof(uint16_t);

    // 匯入：uint16_t 陣列 → mpz
    mpz_import(a, n_limbs, 1, sizeof(uint16_t), 0, 0, in1);
    mpz_import(b, n_limbs, 1, sizeof(uint16_t), 0, 0, in2);

    // 相乘
    mpz_mul(result, a, b);

    // 匯出為 uint16_t 陣列（最大長度 2N limbs）
    size_t count;
    mpz_export(out, &count, 1, sizeof(uint16_t), 0, 0, result);

    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(result);
}