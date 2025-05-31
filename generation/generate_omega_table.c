#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

// === 模數 ===
#define P 2013265921ULL
#define N 4096

// === 普通模乘 ===
static inline uint64_t modmul(uint64_t a, uint64_t b, uint64_t mod) {
    return (__uint128_t)a * b % mod;
}

// === 模指數（右到左平方法）===
uint64_t modpow(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1)
            result = modmul(result, base, mod);
        base = modmul(base, base, mod);
        exp >>= 1;
    }
    return result;
}

// === 模反元素（用 Fermat 小定理）===
uint64_t modinv(uint64_t x, uint64_t p) {
    return modpow(x, p - 2, p);
}

// === 計算 R^2 mod p，R = 2^64 ===
uint64_t compute_R2_mod_p(uint64_t p) {
    __uint128_t r = (__uint128_t)1 << 64;
    uint64_t r_mod_p = (uint64_t)(r % p);
    return modmul(r_mod_p, r_mod_p, p);
}

// === 動態計算 NINV，並正確執行 Montgomery Reduction ===
static inline uint64_t montmul(uint64_t a, uint64_t b, uint64_t ninv, uint64_t p) {
    __uint128_t t = (__uint128_t)a * b;
    uint64_t m = (uint64_t)t * ninv;
    __uint128_t u = t + (__uint128_t)m * p;
    u >>= 64;
    return (uint64_t)u >= p ? (uint64_t)u - p : (uint64_t)u;
}

uint64_t compute_ninv(uint64_t p) {
    uint64_t inv = 1;
    for (int i = 0; i < 5; i++) {
        inv *= 2 - p * inv; // Newton-Raphson approximation
    }
    return (uint64_t)(0 - inv);  // NINV = -P^{-1} mod 2^64
}

int main() {
    uint64_t omega = modpow(31, (P - 1) / N, P);       // ω = 31^((P−1)/N) mod P
    uint64_t R2_mod_p = compute_R2_mod_p(P);
    uint64_t ninv = compute_ninv(P);
    printf("// P = %" PRIu64 ", ω = %" PRIu64 ", R2_mod_p = %" PRIu64 ", NINV = %" PRIu64 "\n", P, omega, R2_mod_p, ninv);

    printf("static const uint64_t omega_table[%d] = {\n    ", N);

    uint64_t w = 1;
    /*for (int i = 0; i < N; i++) {
        uint64_t mont_w = montmul(w, R2_mod_p, ninv, P);  // 轉進 Montgomery domain
        printf("%" PRIu64 "ULL", mont_w);
        if (i != N - 1) printf(", ");
        if ((i + 1) % 4 == 0) printf("\n    ");
        w = modmul(w, omega, P);
    }*/

    printf("\n};\n");
    return 0;
}