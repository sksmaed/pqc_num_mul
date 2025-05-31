#ifndef FNT_UTILS_H
#define FNT_UTILS_H

#include <stdint.h>

// fnt_forward: a = FNT(a) mod P
void fnt_forward(uint64_t *a, const uint64_t *omega_table, int n);

// fnt_inverse: a = FNT^{-1}(a) mod P
void fnt_inverse(uint64_t *a, const uint64_t *omega_inv_table, int n);

// reverse_bits: reverse the bits of x in log_n bits
uint32_t reverse_bits(uint32_t x, int log_n);

// bit-reverse copy
void bit_reverse_copy(uint64_t *dst, const uint64_t *src, int n);

// normalize the array by dividing each element by n mod p
void normalize(uint64_t *a, int n, uint64_t p);

void fnt_forward_fixed_unrolled4096(uint64_t *a);
#endif
