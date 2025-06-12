#ifndef MUL_H
#define MUL_H

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

typedef uint16_t limb_t;

// fnt_mul: c = a * b mod P (convolution-style)
void gmp_mul(size_t n, const void *in1, const void *in2, void *out);
void toom3_mul(limb_t *result, const limb_t *a, const limb_t *b);
void toom4_mul(limb_t *result, const limb_t *a, const limb_t *b); 

#endif
