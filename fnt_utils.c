#include "../headers/constant.h"
#include "../headers/fnt_utils.h"
#include "../headers/utils.h"
#include <stdint.h>

void fnt_forward(uint64_t *a, const uint64_t *omega_table, int n) {
    for (int s = 0; s < STAGES; ++s) {
        int m = 1 << s;           // size of sub-FFT block / 2
        int stride = n >> (s + 1);  // ω 表的跳躍間距

        for (int k = 0; k < n; k += 2 * m) {
            for (int j = 0; j < m; ++j) {
                int idx1 = k + j;
                int idx2 = k + j + m;

                uint64_t t1 = a[idx1];
                uint64_t t2 = modmul(a[idx2], omega_table[j * stride], P);

                a[idx1] = modadd(t1, t2, P);
                a[idx2] = modsub(t1, t2, P);
            }
        }
    }
}

uint32_t reverse_bits(uint32_t x, int log_n) {
    uint32_t r = 0;
    for (int i = 0; i < log_n; ++i) {
        r <<= 1;
        r |= (x & 1);
        x >>= 1;
    }
    return r;
}

void bit_reverse_copy(uint64_t *dst, const uint64_t *src, int n) {
    int log_n = 0;
    while ((1 << log_n) < n) log_n++;

    for (int i = 0; i < n; ++i) {
        uint32_t rev = reverse_bits(i, log_n);
        dst[rev] = src[i];
    }
}

void fnt_inverse(uint64_t *a, const uint64_t *omega_inv_table, int n) {
    for (int s = 0; s < STAGES; ++s) {
        int m = 1 << s;
        int stride = n >> (s + 1);

        for (int k = 0; k < n; k += 2 * m) {
            for (int j = 0; j < m; ++j) {
                int idx1 = k + j;
                int idx2 = k + j + m;

                uint64_t t1 = a[idx1];
                uint64_t t2 = modmul(a[idx2], omega_inv_table[j * stride], P);

                a[idx1] = modadd(t1, t2, P);
                a[idx2] = modsub(t1, t2, P);
            }
        }
    }
}

void normalize(uint64_t *a, int n, uint64_t p) {
    uint64_t inv_n = modinv(n, p);
    for (int i = 0; i < n; ++i) {
        a[i] = modmul(a[i], inv_n, p);
    }
}