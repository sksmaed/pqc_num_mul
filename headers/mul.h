#ifndef MUL_H
#define MUL_H

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

typedef uint16_t limb_t;

// fnt_mul: c = a * b mod P (convolution-style)
void fnt_mul(uint64_t *restrict output,
            const uint64_t *restrict a,
            const uint64_t *restrict b);
void fnt_mul_2048limb(uint64_t *out, const uint64_t *a, const uint64_t *b);
void gmp_mul(size_t n, const void *in1, const void *in2, void *out);
void toom3_mul_2048(limb_t *result, const limb_t *a, const limb_t *b); 

#endif
