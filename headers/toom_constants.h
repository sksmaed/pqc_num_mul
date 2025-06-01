# ifndef TOOM_CONSTANTS_H
# define TOOM_CONSTANTS_H

#include <stdio.h>
#include <inttypes.h>

#define P         12289
#define R         65536
#define BARRETT_V ((1U << 16) / P)
#define DINV      10241 // D = 6
#define MU_DINV   27307

#define LIMB_PER_PART 128
#define HALF 64
#define PROD_LIMBS 256
#define TOTAL_LIMBS   384
#define NUM_PARTS     3
#define NUM_EVALS     5
#define RESULT_LIMBS  768
#define COEFFS        5
#define LIMB_SIZE      16
#define BASE           (1 << LIMB_SIZE)
#define LIMB_BITS      16
#define HALF           64
#define QUARTER        32

const uint16_t x_powers[5][5] = {
    {1, -1, 1, -1, 1},
    {1, 0, 0, 0, 0},
    {1, 1, 1, 1, 1},
    {1, 2, 4, 8, 16},
    {0, 0, 0, 0, 1}
};

static const uint16_t inv_mat[5][5] = {
    { 0, 6, 0, 0, 0 },
    { 12287, 12286, 6, 12288, 12 },
    { 3, 12283, 3, 0, 12283 },
    { 12288, 3, 12286, 1, 12277 },
    { 0, 0, 0, 0, 6 },
};

static const uint16_t inv_mu[5][5] = {
    { 0, 15, 0, 0, 0 },
    { 32762, 32760, 15, 32765, 31 },
    { 7, 32752, 7, 0, 32752 },
    { 32765, 7, 32760, 2, 32736 },
    { 0, 0, 0, 0, 15 },
};

#endif // TOOM_CONSTANTS_H