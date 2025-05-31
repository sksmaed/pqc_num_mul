#include <stdint.h>
#include <stdlib.h>
#include "../headers/constant.h"
#include "../headers/utils.h"

uint64_t rand64_modp(void) {
    return ((uint64_t)rand() << 32 | rand()) % P;
}

uint64_t modmul(uint64_t a, uint64_t b, uint64_t p) {
    __uint128_t result = (__uint128_t)a * b;
    return (uint64_t)(result % p);
}

uint64_t modpow(uint64_t base, uint64_t exp, uint64_t p) {
    uint64_t result = 1;
    base %= p;
    while (exp) {
        if (exp & 1)
            result = modmul(result, base, p);
        base = modmul(base, base, p);
        exp >>= 1;
    }
    return result;
}

uint64_t modadd(uint64_t a, uint64_t b, uint64_t p) {
    uint64_t res = a + b;
    return (res >= p) ? res - p : res;
}

uint64_t modsub(uint64_t a, uint64_t b, uint64_t p) {
    return (a >= b) ? a - b : p + a - b;
}

uint64_t modinv(uint64_t x, uint64_t p) {
    return modpow(x, p - 2, p); 
}