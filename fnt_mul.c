#include "../headers/constant.h"
#include "../headers/fnt_utils.h"
#include "../headers/utils.h"
#include "../headers/omega_table.h"
#include "../headers/omega_inv_table.h"
#include "../headers/mul.h"
#include <stdio.h>

void fnt_mul(uint64_t *restrict output, const uint64_t *restrict a, const uint64_t *restrict b) {
    uint64_t A[N], B[N], C[N];

    // Step 1: copy & bit-reverse
    bit_reverse_copy(A, a, N);
    bit_reverse_copy(B, b, N);

    // Step 2: forward transform
    fnt_forward(A, omega_table, N);
    fnt_forward(B, omega_table, N);

    // Step 3: point-wise multiply
    for (int i = 0; i < N; i++) {
        C[i] = modmul(A[i], B[i], P);
    }

    // Step 4: inverse transform
    fnt_inverse(C, omega_inv_table, N);

    // Step 5: normalize by N^-1 mod P
    normalize(C, N, P);

    // Output
    for (int i = 0; i < N; i++) {
        output[i] = C[i];
    }
}