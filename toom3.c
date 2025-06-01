#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <arm_neon.h>
#include <assert.h>
#include "../headers/toom_constants.h"

typedef uint16_t limb_t;

static inline void split(limb_t out[NUM_PARTS][LIMB_PER_PART], const limb_t input[TOTAL_LIMBS]);
void evaluate_at_points(limb_t eval[NUM_EVALS][LIMB_PER_PART], limb_t parts[NUM_PARTS][LIMB_PER_PART]);
static inline uint16x8_t barrett_reduce_neon_u32(uint32x4_t b_lo, uint32x4_t b_hi);
void karatsuba_mul128(const limb_t *a, const limb_t *b, uint32_t *res);
void karatsuba_mul64(const limb_t *a, const limb_t *b, uint32_t *res);
void modular_multiply(limb_t prod[NUM_EVALS][PROD_LIMBS], limb_t a_eval[NUM_EVALS][LIMB_PER_PART], limb_t b_eval[NUM_EVALS][LIMB_PER_PART]);
void interpolate_at_coeffs(limb_t coeffs[COEFFS][PROD_LIMBS], limb_t prod[NUM_EVALS][PROD_LIMBS]);
void recompose_final_result(limb_t *result, limb_t coeffs[COEFFS][PROD_LIMBS], size_t limb_offset);
static void fix_carry_u16(limb_t *res, size_t N);
void toom3_mul_2048(limb_t *result, const limb_t *a, const limb_t *b);

static inline void split(limb_t out[NUM_PARTS][LIMB_PER_PART], const limb_t input[TOTAL_LIMBS]) {
    for (size_t i = 0; i < NUM_PARTS; ++i) {
        for (size_t j = 0; j < LIMB_PER_PART; ++j) {
            size_t idx = i * LIMB_PER_PART + j;
            out[i][j] = input[idx];
        }
    }
}

static inline uint16x8_t barrett_reduce_neon_u32(uint32x4_t b_lo, uint32x4_t b_hi) {
    const uint32_t V = ((1ULL << 32) + P / 2) / P;
    const uint32x4_t MOD = vdupq_n_u32(P);

    uint32_t b_lo_arr[4], b_hi_arr[4];
    vst1q_u32(b_lo_arr, b_lo);
    vst1q_u32(b_hi_arr, b_hi);

    uint32_t q_lo_arr[4], q_hi_arr[4];

    for (int i = 0; i < 4; ++i) {
        uint64_t prod = (uint64_t)b_lo_arr[i] * V;
        q_lo_arr[i] = (uint32_t)(prod >> 32);

        prod = (uint64_t)b_hi_arr[i] * V;
        q_hi_arr[i] = (uint32_t)(prod >> 32);
    }

    uint32x4_t q_lo = vld1q_u32(q_lo_arr);
    uint32x4_t q_hi = vld1q_u32(q_hi_arr);

    uint32x4_t z_lo = vsubq_u32(b_lo, vmulq_u32(q_lo, MOD));
    uint32x4_t z_hi = vsubq_u32(b_hi, vmulq_u32(q_hi, MOD));

    // if z >= P then z -= P
    uint32x4_t mask_lo = vcgeq_u32(z_lo, MOD);
    z_lo = vsubq_u32(z_lo, vandq_u32(mask_lo, MOD));

    uint32x4_t mask_hi = vcgeq_u32(z_hi, MOD);
    z_hi = vsubq_u32(z_hi, vandq_u32(mask_hi, MOD));

    return vcombine_u16(vqmovn_u32(z_lo), vqmovn_u32(z_hi));
}

void evaluate_at_points(
    limb_t eval[NUM_EVALS][LIMB_PER_PART],
    limb_t parts[NUM_PARTS][LIMB_PER_PART]
) {
    const uint16x8_t P_vec = vdupq_n_u16(P);

    for (size_t i = 0; i < LIMB_PER_PART; i += 8) {
        uint16x8_t a0 = vld1q_u16(&parts[0][i]);
        uint16x8_t a1 = vld1q_u16(&parts[1][i]);
        uint16x8_t a2 = vld1q_u16(&parts[2][i]);

        // x = -1 → a0 - a1 + a2
        uint16x8_t neg_a1 = vsubq_u16(P_vec, a1);  // a1 × (-1) mod P
        uint16x8_t v = vaddq_u16(a0, vaddq_u16(neg_a1, a2));
        v = vsubq_u16(v, vandq_u16(vcgeq_u16(v, P_vec), P_vec));
        vst1q_u16(&eval[0][i], v);

        // x = 0 → a0
        vst1q_u16(&eval[1][i], a0);

        // x = 1 → a0 + a1 + a2
        v = vaddq_u16(a0, vaddq_u16(a1, a2));
        v = vsubq_u16(v, vandq_u16(vcgeq_u16(v, P_vec), P_vec));
        vst1q_u16(&eval[2][i], v);

        // x = 2 → a0 + 2a1 + 4a2
        uint16x8_t a1_2 = vshlq_n_u16(a1, 1);  // a1 × 2
        uint16x8_t a2_4 = vshlq_n_u16(a2, 2);  // a2 × 4
        v = vaddq_u16(a0, vaddq_u16(a1_2, a2_4));
        v = vsubq_u16(v, vandq_u16(vcgeq_u16(v, P_vec), P_vec));
        vst1q_u16(&eval[3][i], v);

        // x = ∞ → a2
        vst1q_u16(&eval[4][i], a2);
    }
}

void karatsuba_mul64(const limb_t *a, const limb_t *b, uint32_t *res) {
    const limb_t *a0 = a;
    const limb_t *a1 = a + QUARTER;
    const limb_t *b0 = b;
    const limb_t *b1 = b + QUARTER;

    uint32_t *z0 = res;
    uint32_t *z2 = res + HALF;
    uint32_t z1[HALF] = {0};

    limb_t a_sum[QUARTER];
    limb_t b_sum[QUARTER];

    uint16x8_t vec_u16_a, vec_u16_b;
    uint32x4_t acc_lo, acc_hi, tmp_lo, tmp_hi;
    uint32_t *res_ptr;

    // a_sum = a0 + a1, b_sum = b0 + b1
    for (size_t i = 0; i < QUARTER; i += 8) {
        vec_u16_a = vaddq_u16(vld1q_u16(&a0[i]), vld1q_u16(&a1[i]));
        vec_u16_b = vaddq_u16(vld1q_u16(&b0[i]), vld1q_u16(&b1[i]));
        vst1q_u16(&a_sum[i], vec_u16_a);
        vst1q_u16(&b_sum[i], vec_u16_b);
    }

    // z0 = a0 * b0
    for (size_t i = 0; i < QUARTER; ++i) {
        vec_u16_a = vdupq_n_u16(a0[i]);
        for (size_t j = 0; j < QUARTER; j += 8) {
            vec_u16_b = vld1q_u16(&b0[j]);
            res_ptr = &z0[i + j];

            acc_lo = vmull_u16(vget_low_u16(vec_u16_a), vget_low_u16(vec_u16_b));
            tmp_lo = vld1q_u32(res_ptr);
            vst1q_u32(res_ptr, vaddq_u32(tmp_lo, acc_lo));

            acc_hi = vmull_u16(vget_high_u16(vec_u16_a), vget_high_u16(vec_u16_b));
            tmp_hi = vld1q_u32(res_ptr + 4);
            vst1q_u32(res_ptr + 4, vaddq_u32(tmp_hi, acc_hi));
        }
    }

    // z2 = a1 * b1
    for (size_t i = 0; i < QUARTER; ++i) {
        vec_u16_a = vdupq_n_u16(a1[i]);
        for (size_t j = 0; j < QUARTER; j += 8) {
            vec_u16_b = vld1q_u16(&b1[j]);
            res_ptr = &z2[i + j];

            acc_lo = vmull_u16(vget_low_u16(vec_u16_a), vget_low_u16(vec_u16_b));
            tmp_lo = vld1q_u32(res_ptr);
            vst1q_u32(res_ptr, vaddq_u32(tmp_lo, acc_lo));

            acc_hi = vmull_u16(vget_high_u16(vec_u16_a), vget_high_u16(vec_u16_b));
            tmp_hi = vld1q_u32(res_ptr + 4);
            vst1q_u32(res_ptr + 4, vaddq_u32(tmp_hi, acc_hi));
        }
    }

    // z1 = (a0+a1)*(b0+b1)
    for (size_t i = 0; i < QUARTER; ++i) {
        vec_u16_a = vdupq_n_u16(a_sum[i]);
        for (size_t j = 0; j < QUARTER; j += 8) {
            vec_u16_b = vld1q_u16(&b_sum[j]);
            res_ptr = &z1[i + j];

            acc_lo = vmull_u16(vget_low_u16(vec_u16_a), vget_low_u16(vec_u16_b));
            tmp_lo = vld1q_u32(res_ptr);
            vst1q_u32(res_ptr, vaddq_u32(tmp_lo, acc_lo));

            acc_hi = vmull_u16(vget_high_u16(vec_u16_a), vget_high_u16(vec_u16_b));
            tmp_hi = vld1q_u32(res_ptr + 4);
            vst1q_u32(res_ptr + 4, vaddq_u32(tmp_hi, acc_hi));
        }
    }

    // z1 = z1 - z0 - z2, accumulate to res[QUARTER + i]
    for (size_t i = 0; i < HALF; i += 4) {
        acc_lo = vld1q_u32(&z1[i]);
        tmp_lo = vld1q_u32(&z0[i]);
        tmp_hi = vld1q_u32(&z2[i]);
        acc_lo = vsubq_u32(acc_lo, vaddq_u32(tmp_lo, tmp_hi));

        tmp_lo = vld1q_u32(&res[QUARTER + i]);
        vst1q_u32(&res[QUARTER + i], vaddq_u32(tmp_lo, acc_lo));
    }
}

void karatsuba_mul128(const limb_t *a, const limb_t *b, uint32_t *res) {
    const limb_t *a0 = a;
    const limb_t *a1 = a + HALF;
    const limb_t *b0 = b;
    const limb_t *b1 = b + HALF;

    uint32_t *z0 = res;                        // 0..127
    uint32_t *z2 = res + LIMB_PER_PART;        // 128..255
    uint32_t z1[LIMB_PER_PART] = {0};          // 128 limbs

    limb_t a_sum[HALF] __attribute__((aligned(16))) = {0};
    limb_t b_sum[HALF] __attribute__((aligned(16))) = {0};

    // Vectorized addition: a_sum = a0 + a1, b_sum = b0 + b1
    for (size_t i = 0; i < HALF; i += 8) {
        vst1q_u16(&a_sum[i], vaddq_u16(vld1q_u16(&a0[i]), vld1q_u16(&a1[i])));
        vst1q_u16(&b_sum[i], vaddq_u16(vld1q_u16(&b0[i]), vld1q_u16(&b1[i])));
    }

    // Recursive step: z0 = a0 * b0, z2 = a1 * b1, z1 = (a0+a1)*(b0+b1)
    karatsuba_mul64(a0, b0, z0);
    karatsuba_mul64(a1, b1, z2);
    karatsuba_mul64(a_sum, b_sum, z1);

    // z1 -= z0 + z2
    for (size_t i = 0; i < LIMB_PER_PART; ++i) {
        res[HALF + i] += z1[i] - z0[i] - z2[i];
    }
}

void modular_multiply(
    limb_t prod[NUM_EVALS][PROD_LIMBS],
    limb_t a_eval[NUM_EVALS][LIMB_PER_PART],
    limb_t b_eval[NUM_EVALS][LIMB_PER_PART]
) {
    for (int i = 0; i < NUM_EVALS; ++i) {
        uint32_t tmp[PROD_LIMBS] = {0};  // local buffer for result
        karatsuba_mul128(a_eval[i], b_eval[i], tmp);
        
        // reduce tmp -> prod[i]
        for (size_t j = 0; j < PROD_LIMBS; j += 8) {
            uint32x4_t lo = vld1q_u32(&tmp[j]);
            uint32x4_t hi = vld1q_u32(&tmp[j + 4]);
            
            uint16x8_t reduced = barrett_reduce_neon_u32(lo, hi);
            vst1q_u16(&prod[i][j], reduced);
        }
    }
}

void interpolate_at_coeffs(
    limb_t coeffs[COEFFS][PROD_LIMBS],
    limb_t prod[NUM_EVALS][PROD_LIMBS]
) {
    const uint32x4_t P_vec = vdupq_n_u32(P);

    uint32x4_t acc, q, r, r_sub_p, mask;

    for (size_t limb = 0; limb < PROD_LIMBS; limb += 4) {
        // Load 5 evaluation results (4 limbs each)
        uint16x4_t w[NUM_EVALS];
        for (int j = 0; j < NUM_EVALS; ++j)
            w[j] = vld1_u16(&prod[j][limb]);

        for (int i = 0; i < COEFFS; ++i) {
            // 32-bit accumulation to avoid overflow
            acc = vmull_n_u16(w[0], inv_mu[i][0]);
            for (int j = 1; j < NUM_EVALS; ++j)
                acc = vmlal_n_u16(acc, w[j], inv_mu[i][j]);
            
            // Barrett reduciton
            q = vshrq_n_u32(vmulq_n_u32(acc, BARRETT_V), 16);
            r = vsubq_u32(acc, vmulq_n_u32(q, P));

            // if r >= P then r -= P
            r_sub_p = vsubq_u32(r, P_vec);
            mask = vcgeq_u32(r, P_vec);
            r = vbslq_u32(mask, r_sub_p, r);
            
            // Save result (narrow to 16-bit)
            vst1_u16(&coeffs[i][limb], vmovn_u32(r));
        }
    }
}

static void fix_carry_u16(limb_t *res, size_t N) {
    uint32_t carry = 0;
    for (size_t i = 0; i < N; ++i) {
        uint32_t sum = (uint32_t)res[i] + carry;
        res[i] = (limb_t)(sum & 0xFFFF);
        carry = sum >> 16;
    }
}

void recompose_final_result(limb_t *result, limb_t coeffs[COEFFS][PROD_LIMBS], size_t limb_offset) {
    memset(result, 0, sizeof(limb_t) * RESULT_LIMBS);

    // SIMD-based add (no carry)
    for (int i = 0; i < COEFFS; ++i) {
        size_t shift = i * limb_offset;
        for (size_t j = 0; j < PROD_LIMBS; j += 8) {
            uint16x8_t r = vld1q_u16(&result[shift + j]);
            uint16x8_t c = vld1q_u16(&coeffs[i][j]);
            uint16x8_t s = vaddq_u16(r, c);
            vst1q_u16(&result[shift + j], s);
        }
    }

    // scalar carry fix pass
    fix_carry_u16(result, RESULT_LIMBS);
}

void toom3_mul_2048(limb_t *result, const limb_t *a, const limb_t *b) {
    // 1. Split into 3 parts
    limb_t a_parts[NUM_PARTS][LIMB_PER_PART] __attribute__((aligned(16)));
    limb_t b_parts[NUM_PARTS][LIMB_PER_PART] __attribute__((aligned(16)));
    split(a_parts, a);
    split(b_parts, b);

    // 2. Evaluate at x ∈ {-1, 0, 1, 2, infty}
    limb_t a_eval[NUM_EVALS][LIMB_PER_PART] __attribute__((aligned(16))) = {0};
    limb_t b_eval[NUM_EVALS][LIMB_PER_PART] __attribute__((aligned(16))) = {0};
    evaluate_at_points(a_eval, a_parts);
    evaluate_at_points(b_eval, b_parts);

    // 3. Modular multiply A(x_i) * B(x_i)
    limb_t prod[NUM_EVALS][PROD_LIMBS] = {0};
    modular_multiply(prod, a_eval, b_eval);

    // 4. Interpolate to get coefficients
    limb_t coeffs[COEFFS][PROD_LIMBS] = {0};
    interpolate_at_coeffs(coeffs, prod);

    // 5. Compose final result with shifting and adding
    recompose_final_result(result, coeffs, LIMB_PER_PART);  // limb_offset = 256
}