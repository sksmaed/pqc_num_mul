#include <arm_neon.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include "../headers/fnt_utils.h"
#include "../headers/omega_table.h"
#include "../headers/omega_inv_table.h"
#include "../headers/constant.h"
#include "../headers/utils.h"
#include "../headers/mul.h"

#define NINV 14393504411089371135ULL // -P^{-1} mod 2^64

static inline uint64_t reduce128(__uint128_t x) {
    uint64_t res = (uint64_t)(x >> 64);
    return (res >= P) ? res - P : res;
}

static inline uint64_t mul_mont_domain(uint64_t a, uint64_t b) {
    return reduce128((__uint128_t)a * b);
}

static inline uint64_t montmul(uint64_t a, uint64_t b) {
    __uint128_t t = (__uint128_t)a * b;
    uint64_t m = (uint64_t)t * NINV;
    __uint128_t u = (t + (__uint128_t)m * P) >> 64;
    return (uint64_t)u - ((uint64_t)u >= P ? P : 0);
}

/*static inline uint64_t montmul(uint64_t a, uint64_t b) {
    uint64_t lo, hi, m, t_hi, u;

    __asm__ volatile (
        // t = a * b = [hi:lo]
        "mul    %[lo], %[a], %[b]\n\t"
        "umulh  %[hi], %[a], %[b]\n\t"

        // m = lo * NINV
        "mul    %[m], %[lo], %[ninv]\n\t"

        // t_hi = high(m * P)
        "umulh  %[t_hi], %[m], %[p]\n\t"

        // u = hi + t_hi
        "add    %[u], %[hi], %[t_hi]\n\t"
        "cmp    %[u], %[p]\n\t"
        "csel   %[u], %[u], xzr, lo\n\t"   // if u >= P, keep u; else keep u
    : [lo] "=&r"(lo),
      [hi] "=&r"(hi),
      [m] "=&r"(m),
      [t_hi] "=&r"(t_hi),
      [u] "=&r"(u)
    : [a] "r"(a),
      [b] "r"(b),
      [ninv] "r"(NINV),
      [p] "r"(P)
    : "cc"
    );

    return (u >= P) ? u - P : u;
}*/

static inline uint64_t modadd_lazy(uint64_t a, uint64_t b) {
    return a + b;
}

static inline uint64_t modsub_lazy(uint64_t a, uint64_t b, uint64_t p) {
    return (a >= b) ? a - b : a + p - b;
}

static inline void bit_reversal_swap(uint64_t *a, int n) {
    int log_n = 0;
    while ((1 << log_n) < n) ++log_n;

    for (int i = 0; i < n; ++i) {
        int j = reverse_bits(i, log_n);
        if (j > i) {
            uint64_t tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }
    }
}

static inline void pointwise_mul_inplace(uint64_t *out, const uint64_t *restrict a, const uint64_t *restrict b, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = montmul(a[i], b[i]);
    }
}

static inline void fnt_butterfly_generic(
    uint64_t *a,
    const uint64_t *table,
    int log4_N) 
{
    for (int s = 0; s < log4_N; ++s) {
        int m = 1 << (2 * s);
        int stride = N / (4 * m);

        for (int k = 0; k < N; k += 4 * m) {
            for (int j = 0; j < m; j += 2) {
                if (j + 1 >= m) continue;

                int i0  = k + j;
                int i0b = k + j + 1;
                int i1  = i0  + m;
                int i1b = i0b + m;
                int i2  = i1  + m;
                int i2b = i1b + m;
                int i3  = i2  + m;
                int i3b = i2b + m;

                if (i3b >= N) continue;

                int w1_idx = 1 * j * stride;
                int w2_idx = 2 * j * stride;
                int w3_idx = 3 * j * stride;

                int w1b_idx = 1 * (j + 1) * stride;
                int w2b_idx = 2 * (j + 1) * stride;
                int w3b_idx = 3 * (j + 1) * stride;

                uint64_t w1a = table[w1_idx];
                uint64_t w2a = table[w2_idx];
                uint64_t w3a = table[w3_idx];

                uint64_t w1b = table[w1b_idx];
                uint64_t w2b = table[w2b_idx];
                uint64_t w3b = table[w3b_idx];

                uint64x2_t a0 = {a[i0], a[i0b]};
                uint64x2_t a1 = {a[i1], a[i1b]};
                uint64x2_t a2 = {a[i2], a[i2b]};
                uint64x2_t a3 = {a[i3], a[i3b]};

                uint64x2_t t0 = {modadd_lazy(a0[0], a2[0]), modadd_lazy(a0[1], a2[1])};
                uint64x2_t t1 = {modsub_lazy(a0[0], a2[0], P), modsub_lazy(a0[1], a2[1], P)};
                uint64x2_t t2 = {modadd_lazy(a1[0], a3[0]), modadd_lazy(a1[1], a3[1])};
                uint64x2_t t3 = {modsub_lazy(a1[0], a3[0], P), modsub_lazy(a1[1], a3[1], P)};

                a[i0]  = modadd_lazy(t0[0], t2[0]);
                a[i0b] = modadd_lazy(t0[1], t2[1]);

                a[i1]  = mul_mont_domain(modadd_lazy(t1[0], t3[0]), w1a);
                a[i1b] = mul_mont_domain(modadd_lazy(t1[1], t3[1]), w1b);

                a[i2]  = mul_mont_domain(modsub_lazy(t0[0], t2[0], P), w2a);
                a[i2b] = mul_mont_domain(modsub_lazy(t0[1], t2[1], P), w2b);

                a[i3]  = mul_mont_domain(modsub_lazy(t1[0], t3[0], P), w3a);
                a[i3b] = mul_mont_domain(modsub_lazy(t1[1], t3[1], P), w3b);
            }
        }
    }
}

/*static inline void fnt_forward_fixed(uint64_t *a) {
    fnt_butterfly_generic(a, omega_table, 6);
}

static inline void fnt_inverse_fixed(uint64_t *a) {
    fnt_butterfly_generic(a, omega_inv_table, 6);
}*/

static inline void fnt_forward_fixed(uint64_t *a) {
    int log4_N = 6;
    for (int s = 0; s < log4_N; ++s) {
        int m = 1 << (2 * s);
        int stride = N / (4 * m);

        for (int k = 0; k < N; k += 4 * m) {
            for (int j = 0; j < m; j++) {
                int i0  = k + j;
                int i1  = i0 + m;
                int i2  = i1 + m;
                int i3  = i2 + m;

                if (i3 >= N) continue;

                uint64_t w1 = omega_table[1 * j * stride];
                uint64_t w2 = omega_table[2 * j * stride];
                uint64_t w3 = omega_table[3 * j * stride];

                uint64_t a0 = a[i0];
                uint64_t a1 = mul_mont_domain(a[i1], w1);
                uint64_t a2 = mul_mont_domain(a[i2], w2);
                uint64_t a3 = mul_mont_domain(a[i3], w3);

                uint64_t t0 = modadd_lazy(a0, a2);
                uint64_t t1 = modsub_lazy(a0, a2, P);
                uint64_t t2 = modadd_lazy(a1, a3);
                uint64_t t3 = modsub_lazy(a1, a3, P);

                a[i0] = modadd_lazy(t0, t2);
                a[i1] = montmul(modadd_lazy(t1, t3), w1);
                a[i2] = montmul(modsub_lazy(t0, t2, P), w2);
                a[i3] = montmul(modsub_lazy(t1, t3, P), w3);
            }
        }
    }
}

static inline void fnt_inverse_fixed(uint64_t *a) {
    int log4_N = 6;
    for (int s = 0; s < log4_N; ++s) {
        int m = 1 << (2 * s);
        int stride = N / (4 * m);

        for (int k = 0; k < N; k += 4 * m) {
            for (int j = 0; j < m; ++j) {
                int i0 = k + j;
                int i1 = i0 + m;
                int i2 = i1 + m;
                int i3 = i2 + m;

                if (i3 >= N) continue;

                uint64_t w1 = omega_inv_table[1 * j * stride];
                uint64_t w2 = omega_inv_table[2 * j * stride];
                uint64_t w3 = omega_inv_table[3 * j * stride];

                uint64_t a0 = a[i0];
                uint64_t a1 = a[i1];
                uint64_t a2 = a[i2];
                uint64_t a3 = a[i3];

                uint64_t t0 = modadd_lazy(a0, a2);
                uint64_t t1 = modsub_lazy(a0, a2, P);
                uint64_t t2 = modadd_lazy(a1, a3);
                uint64_t t3 = modsub_lazy(a1, a3, P);

                a[i0] = modadd_lazy(t0, t2);
                a[i1] = montmul(modadd_lazy(t1, t3), w1);
                a[i2] = montmul(modsub_lazy(t0, t2, P), w2);
                a[i3] = montmul(modsub_lazy(t1, t3, P), w3);
            }
        }
    }
}

void fnt_mul_2048limb(uint64_t *out, const uint64_t *a, const uint64_t *b) {
    uint64_t A[N], B[N];

    for (int i = 0; i < 2048; i++) {
    A[bitrev_table[i]] = montmul(a[i], R2_mod_p);
    B[bitrev_table[i]] = montmul(b[i], R2_mod_p);
}
    for (int i = 2048; i < N; i++) {
        A[bitrev_table[i]] = 0;
        B[bitrev_table[i]] = 0;
    }

    fnt_forward_fixed(A);
    fnt_forward_fixed(B);
    /*for (int i = 0; i < N; ++i) {
        printf("A[%d] = %" PRIu64 "\n", i, A[i]);
    }*/
    pointwise_mul_inplace(A, A, B, N);
    fnt_inverse_fixed(A);
    
    for (int i = 0; i < N; ++i) {
        out[i] = montmul(A[i], 1);
        // printf("out[%d] = %" PRIu64 "\n", i, out[i]);
    }
    
    int nonzero = 0;
    for (int i = 0; i < N; ++i) {
        if (out[i] != 0) nonzero++;
    }
    //printf("nonzero out[i] = %d\n", nonzero);
}
