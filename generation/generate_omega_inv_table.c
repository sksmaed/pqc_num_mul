#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

// === 參數 ===
#define P     2013265921ULL
#define N     4096
#define ROOT  31ULL                    // 原根
#define NINV  0xffffffff00000001ULL   // -P^{-1} mod 2^64

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

// === 安全計算 R^2 mod P（避免 overflow）===
uint64_t compute_R2_mod_p(uint64_t p) {
    __uint128_t r = (__uint128_t)1 << 64;
    uint64_t r_mod_p = (uint64_t)(r % p);
    return modmul(r_mod_p, r_mod_p, p);
}

// === Montgomery Reduction ===
static inline uint64_t montmul(uint64_t a, uint64_t b) {
    __uint128_t t = (__uint128_t)a * b;
    uint64_t m = (uint64_t)t * NINV;
    __uint128_t u = t + (__uint128_t)m * P;
    u >>= 64;
    return (uint64_t)u >= P ? (uint64_t)u - P : (uint64_t)u;
}

int main() {
    uint64_t R2_mod_p = compute_R2_mod_p(P);
    uint64_t omega = modpow(ROOT, (P - 1) / N, P);            // ω
    uint64_t omega_inv = modpow(omega, P - 2, P);             // ω⁻¹ using Fermat's little theorem

    printf("// P = %" PRIu64 ", ω = %" PRIu64 ", ω⁻¹ = %" PRIu64 ", R2_mod_p = %" PRIu64 "\n",
           P, omega, omega_inv, R2_mod_p);

    printf("static const uint64_t omega_inv_table[%d] = {\n    ", N);

    uint64_t w = 1;
    for (int i = 0; i < N; i++) {
        uint64_t mont_w = montmul(w, R2_mod_p);  // 將 ω⁻ⁱ 轉進 Montgomery domain
        printf("%" PRIu64 "ULL", mont_w);
        if (i != N - 1) printf(", ");
        if ((i + 1) % 4 == 0) printf("\n    ");
        w = modmul(w, omega_inv, P);
    }

    printf("\n};\n");
    return 0;
}
