#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>
#include <arm_neon.h>
#include <assert.h>
#include <gmp.h>
#include "../headers/toom4_constants.h"
#define vmulq_s64_custom(a, b) \
    (int64x2_t){ (a)[0] * (b)[0], (a)[1] * (b)[1] }

typedef uint16_t limb_t;

static inline void split(limb_t out[NUM_PARTS][LIMB_PER_PART], const limb_t input[TOTAL_LIMBS]);
static inline int32x4_t horner_vec(int32x4_t a0, int32x4_t a1, int32x4_t a2, int32x4_t a3, int32_t x);
static void evaluate_at_points(int32_t out[NUM_EVALS][LIMB_PER_PART], limb_t in [NUM_PARTS][LIMB_PER_PART]);
void karatsuba_mul96_32(const int32_t *a, const int32_t *b, int32_t *res);
void modular_multiply(int32_t prod[NUM_EVALS][PROD_LIMBS], int32_t a_eval[NUM_EVALS][LIMB_PER_PART], int32_t b_eval[NUM_EVALS][LIMB_PER_PART]);
void interpolate_at_coeffs(int32_t coeffs[NUM_EVALS][PROD_LIMBS], int32_t prod[NUM_EVALS][PROD_LIMBS]);
void recompose_final_result(limb_t *result, const int32_t coeffs[NUM_EVALS][PROD_LIMBS], size_t limb_offset);
void toom4_mul(limb_t *result, const limb_t *a, const limb_t *b);
static void karatsuba_rec_32(const int32_t *a, const int32_t *b, size_t n, int64_t *res);
static void schoolbook_mul_24_32(const int32_t *a, const int32_t *b, int64_t *res);
static void carry_fix_32(const int64_t *src, size_t len, int32_t *dst);
static inline int64_t scoeff(uint64_t u);

static inline int64_t scoeff(uint64_t u) {
    return (u > (P >> 1)) ? (int64_t)(u - P) : (int64_t)u;
}

static inline void split(limb_t out[NUM_PARTS][LIMB_PER_PART], const limb_t input[TOTAL_LIMBS]) {
    for (size_t i = 0; i < NUM_PARTS; ++i) {
        for (size_t j = 0; j < LIMB_PER_PART; ++j) {
            size_t idx = i * LIMB_PER_PART + j;
            out[i][j] = input[idx];
        }
    }
}

static inline int32x4_t horner_vec(int32x4_t a0, int32x4_t a1, int32x4_t a2, int32x4_t a3, int32_t x)
{
    int64x2_t X = vdupq_n_s64((int64_t)x);  // 將 scalar x broadcast 成 int64x2_t

    int64x2_t lo = vmull_n_s32(vget_low_s32(a0), x);         // a0 * x
    lo = vaddq_s64(lo, vmovl_s32(vget_low_s32(a1)));         // + a1
    lo = vmulq_s64_custom(lo, X);                                   // * x
    lo = vaddq_s64(lo, vmovl_s32(vget_low_s32(a2)));         // + a2
    lo = vmulq_s64_custom(lo, X);                                   // * x
    lo = vaddq_s64(lo, vmovl_s32(vget_low_s32(a3)));         // + a3

    int64x2_t hi = vmull_n_s32(vget_high_s32(a0), x);
    hi = vaddq_s64(hi, vmovl_s32(vget_high_s32(a1)));
    hi = vmulq_s64_custom(hi, X);
    hi = vaddq_s64(hi, vmovl_s32(vget_high_s32(a2)));
    hi = vmulq_s64_custom(hi, X);
    hi = vaddq_s64(hi, vmovl_s32(vget_high_s32(a3)));

    return vcombine_s32(vqmovn_s64(lo), vqmovn_s64(hi));
}

static void evaluate_at_points(int32_t out[NUM_EVALS][LIMB_PER_PART], limb_t in[NUM_PARTS][LIMB_PER_PART])
{
    // A3 → ∞ (pt = 6)
    for (int i = 0; i < LIMB_PER_PART; ++i)
        out[6][i] = (int32_t)in[3][i];

    // A0 → 0 (pt = 2)
    for (int i = 0; i < LIMB_PER_PART; ++i)
        out[2][i] = (int32_t)in[0][i];

    static const int64_t xs[5] = { -2, -1, 1, 2, 4 };
    static const int     pts[5] = {  0,  1, 3, 4, 5 };

    for (int k = 0; k < 5; ++k) {
        int64_t x = xs[k];
        int pt    = pts[k];

        for (int i = 0; i < LIMB_PER_PART; i += 4) {
            int32_t tmp0[4], tmp1[4], tmp2[4], tmp3[4];
            for (int j = 0; j < 4; ++j) {
                tmp0[j] = (int32_t)in[0][i + j];
                tmp1[j] = (int32_t)in[1][i + j];
                tmp2[j] = (int32_t)in[2][i + j];
                tmp3[j] = (int32_t)in[3][i + j];
            }

            int32x4_t a0 = vld1q_s32(tmp0);
            int32x4_t a1 = vld1q_s32(tmp1);
            int32x4_t a2 = vld1q_s32(tmp2);
            int32x4_t a3 = vld1q_s32(tmp3);
            int32x4_t r;

            if (x == 1) {
                r = vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3));
            } else if (x == -1) {
                r = vsubq_s32(vaddq_s32(a3, a1), vaddq_s32(a2, a0));
            } else {
                r = horner_vec(a0, a1, a2, a3, x);
            }

            vst1q_s32(&out[pt][i], r);
        }
    }
}

static void schoolbook_mul_24_32(const int32_t *a,
                                      const int32_t *b,
                                      int64_t       *res)
{
    memset(res, 0, 48 * sizeof(int64_t));
    for (int ia = 0; ia < 24; ia += 4) {
        int32x4_t   a_vec = vld1q_s32(a + ia);

        for (int ib = 0; ib < 24; ib += 4) {
            int32x4_t b_vec  = vld1q_s32(b + ib);
            int32x2_t b_low  = vget_low_s32 (b_vec);
            int32x2_t b_high = vget_high_s32(b_vec);

            for (int ka = 0; ka < 4; ++ka) {
                int32x2_t a_lane2 = vdup_n_s32(vgetq_lane_s32(a_vec, ka));
                int64x2_t m0 = vmull_s32(a_lane2, b_low);   // × b[ib], b[ib+1]
                int64x2_t m1 = vmull_s32(a_lane2, b_high);  // × b[ib+2], b[ib+3]
                int off = (ia + ka) + ib;
                int64_t *dst = res + off;

                int64x2_t old0 = vld1q_s64(dst);           // res[ off ], res[off+1]
                old0 = vaddq_s64(old0, m0);
                vst1q_s64(dst, old0);
                int64x2_t old1 = vld1q_s64(dst + 2);       // res[ off+2 ], res[off+3]
                old1 = vaddq_s64(old1, m1);
                vst1q_s64(dst + 2, old1);
            }
        }
    }
}

static void karatsuba_rec_32(const int32_t *a, const int32_t *b,
                             size_t n, int64_t *res)
{
    if (n <= 24) { schoolbook_mul_24_32(a, b, res); return; }

    size_t  m  = n >> 1;
    const int32_t *a0 = a, *a1 = a + m;
    const int32_t *b0 = b, *b1 = b + m;

    int64_t z0[2*m], z1[2*m], z2[2*m];
    karatsuba_rec_32(a0, b0, m, z0);
    karatsuba_rec_32(a1, b1, m, z2);

    // z1 = (a0+a1)*(b0+b1)
    memset(z1, 0, sizeof z1);
    for (size_t i = 0; i < m; ++i) {
        int64_t ai = (int64_t)a0[i] + a1[i];
        for (size_t j = 0; j < m; ++j)
            z1[i+j] += ai * ((int64_t)b0[j] + b1[j]);
    }

    __int128 acc[2*n];
    memset(acc, 0, sizeof acc);

    for (size_t k = 0; k < 2*m; ++k) {
        acc[k]           += z0[k];
        acc[k + 2*m]     += z2[k];
        acc[k + m]       += (__int128)z1[k] - z0[k] - z2[k];
    }

    for (size_t k = 0; k < 2*n; ++k) {
        int64_t lo = (int64_t)acc[k];
        int64_t hi = (int64_t)(acc[k] >> 64);
        res[k] = lo;
        if (hi) {
            if (k + 1 < 2*n) res[k + 1] += hi;
        }
    }
}

static void carry_fix_32(const int64_t *src,
                              size_t        len,
                              int32_t      *dst)
{
    int64x2_t carry2 = vdupq_n_s64(0);

    for (size_t i = 0; i < len; i += 2) {
        int64x2_t v  = vld1q_s64(src + i);
        v = vaddq_s64(v, carry2);

        int32x2_t low32  = vmovn_s64(v);
        int32x2_t high32 = vshrn_n_s64(v, 32);

        vst1_s32(dst + i    , low32);
        vst1_lane_s32(dst + i + 1, low32, 1);

        carry2 = vmovl_s32(high32);
    }
}

void karatsuba_mul96_32(const int32_t *a, const int32_t *b, int32_t *res)
{
    int64_t wide[PROD_LIMBS];
    karatsuba_rec_32(a, b, LIMB_PER_PART, wide);
    carry_fix_32(wide, PROD_LIMBS, res);
}

void modular_multiply(
    int32_t prod[NUM_EVALS][PROD_LIMBS],
    int32_t a_eval[NUM_EVALS][LIMB_PER_PART],
    int32_t b_eval[NUM_EVALS][LIMB_PER_PART]
) {
    for (int i = 0; i < NUM_EVALS; ++i) {
        karatsuba_mul96_32(a_eval[i], b_eval[i], prod[i]);
    }
}

void interpolate_at_coeffs(int32_t coeffs[NUM_EVALS][PROD_LIMBS],
                           int32_t prod[NUM_EVALS][PROD_LIMBS])
{
    for (size_t limb = 0; limb < PROD_LIMBS; limb += 4) {
        int32_t w[NUM_EVALS][4];
        for (int j = 0; j < NUM_EVALS; ++j) {
            int32x4_t v = vld1q_s32(&prod[j][limb]);
            vst1q_s32(w[j], v);
        }

        for (int i = 0; i < NUM_EVALS; ++i) {
            __int128 acc[4] = {0};

            for (int j = 0; j < NUM_EVALS; ++j) {
                int64_t coeff = scoeff(inv_mat[i][j]);
                for (int k = 0; k < 4; ++k)
                    acc[k] += (__int128)coeff * w[j][k]; // 128-bit -> prevent overflow
            }

            for (int k = 0; k < 4; ++k) {
                int64_t t  = (int64_t)(acc[k] % (__int128)P);
                if (t < 0) t += P;
                coeffs[i][limb + k] = (int32_t)t;
            }
        }
    }
}

void recompose_final_result(limb_t *result,
                            const int32_t coeffs[NUM_EVALS][PROD_LIMBS],
                            size_t limb_offset)
{
    static int64_t acc[RESULT_LIMBS];
    memset(acc, 0, sizeof acc);

    for (int i = 0; i < 7; ++i){
        size_t base = i * limb_offset;
        for (size_t j = 0; j < 2 * limb_offset; ++j){
            uint32_t w = (uint32_t)coeffs[i][j];
            acc[base + j    ] +=  (int64_t)(w      & 0xFFFF);
            acc[base + j + 1] +=  (int64_t)(w >>16 & 0xFFFF);
        }
    }

    int64_t carry = 0;
    for (size_t k = 0; k < RESULT_LIMBS; ++k) {
        int64_t v = acc[k] + carry;
        int64_t limb = v & 0xFFFF;
        carry        = v >> 16;

        result[k] = (uint16_t)limb;
    }
}

void toom4_mul(limb_t *result, const limb_t *a, const limb_t *b) {
    // 1. Split a, b into 4 parts of 96 limbs each (total = 384 limbs)
    limb_t a_parts[NUM_PARTS][LIMB_PER_PART] __attribute__((aligned(16)));
    limb_t b_parts[NUM_PARTS][LIMB_PER_PART] __attribute__((aligned(16)));
    split(a_parts, a);
    split(b_parts, b);

    // 2. Evaluate a(x), b(x) at x ∈ {-2, -1, 0, 1, 2, 4, ∞}
    int32_t a_eval[NUM_EVALS][LIMB_PER_PART] = {0};
    int32_t b_eval[NUM_EVALS][LIMB_PER_PART] = {0};
    evaluate_at_points(a_eval, a_parts);
    evaluate_at_points(b_eval, b_parts);
    
    // 3. Pointwise modular multiplication (96-limb × 96-limb → 192-limb)
    // Use 32-bit or 64-bit intermediate storage to avoid overflow
    int32_t prod[NUM_EVALS][PROD_LIMBS] = {0};
    modular_multiply(prod, a_eval, b_eval);

    // 4. Interpolation (return 7 × 192-limb polynomial coefficients)
    int32_t coeffs[NUM_EVALS][PROD_LIMBS] = {0};
    interpolate_at_coeffs(coeffs, prod);

    // 5. Recompose final result from coefficients
    recompose_final_result(result, coeffs, LIMB_PER_PART);
}