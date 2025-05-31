# ifndef TOOM_CONSTANTS_H
# define TOOM_CONSTANTS_H

#include <stdio.h>
#include <inttypes.h>

const int K = 6;
const uint16_t x_powers[5][5] = {
    {1, -1, 1, -1, 1},
    {1, 0, 0, 0, 0},
    {1, 1, 1, 1, 1},
    {1, 2, 4, 8, 16},
    {0, 0, 0, 0, 1}
};
const uint16_t P = 12289;
const uint32_t R = 65536;
const uint16_t mu_m1 = 32765;
const uint16_t mu_1 = 2;
const uint16_t mu_2 = 5;

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

// D = 6
static const uint16_t Dinv = 10241;
const uint16_t mu_Dinv = 27307;

#endif // TOOM_CONSTANTS_H