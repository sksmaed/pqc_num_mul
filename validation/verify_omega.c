#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <assert.h>
#include "../headers/omega_inv_table.h"
#include "../headers/omega_table.h"

#define N 4096
#define P 2013265921ULL

const uint64_t omega     = 1282623253ULL;
const uint64_t omega_inv = 1255670133ULL;

// Modular multiplication using __uint128_t
uint64_t modmul(uint64_t a, uint64_t b, uint64_t p) {
    return (uint64_t)((__uint128_t)a * b % p);
}

// Modular exponentiation
uint64_t modpow(uint64_t base, uint64_t exp, uint64_t p) {
    uint64_t result = 1;
    base = base % p;
    while (exp > 0) {
        if (exp & 1)
            result = modmul(result, base, p);
        base = modmul(base, base, p);
        exp >>= 1;
    }
    return result;
}

void verify_omega_basic() {
    printf("▶ Verifying omega^N ≡ 1 mod p...\n");
    assert(modpow(omega, N, P) == 1);
    printf("✅ PASS: omega^N ≡ 1 mod p\n");

    printf("▶ Verifying omega^k ≠ 1 for 0 < k < N...\n");
    for (int k = 1; k < N; k++) {
        uint64_t pow_k = modpow(omega, k, P);
        if (pow_k == 1) {
            printf("❌ FAIL: omega^%d ≡ 1 mod p (not primitive)\n", k);
            assert(0);
        }
    }
    printf("✅ PASS: omega is primitive root of unity\n");

    printf("▶ Verifying omega × omega_inv ≡ 1 mod p...\n");
    assert(modmul(omega, omega_inv, P) == 1);
    printf("✅ PASS: omega × omega_inv ≡ 1 mod p\n");
}

void verify_tables() {
    printf("▶ Verifying omega_table[i] == omega^i mod p...\n");
    for (int i = 0; i < N; i++) {
        uint64_t expected = modpow(omega, i, P);
        if (omega_table[i] != expected) {
            printf("❌ FAIL at i=%d: table=%" PRIu64 ", expected=%" PRIu64 "\n", i, omega_table[i], expected);
            assert(0);
        }
    }
    printf("✅ PASS: omega_table values correct\n");

    printf("▶ Verifying omega_inv_table[i] == omega^-i mod p...\n");
    for (int i = 0; i < N; i++) {
        uint64_t expected = modpow(omega_inv, i, P);
        if (omega_inv_table[i] != expected) {
            printf("❌ FAIL at i=%d: table=%" PRIu64 ", expected=%" PRIu64 "\n", i, omega_inv_table[i], expected);
            assert(0);
        }
    }
    printf("✅ PASS: omega_inv_table values correct\n");

    printf("▶ Verifying omega_table[i] * omega_inv_table[i] ≡ 1 mod p...\n");
    for (int i = 0; i < N; i++) {
        uint64_t prod = modmul(omega_table[i], omega_inv_table[i], P);
        if (prod != 1) {
            printf("❌ FAIL at i=%d: prod=%" PRIu64 " != 1\n", i, prod);
            assert(0);
        }
    }
    printf("✅ PASS: ω^i × ω⁻^i ≡ 1\n");
}

int main() {
    verify_omega_basic();
    verify_tables();
    printf("\n🎉 All ω and ω⁻¹ checks passed successfully!\n");
    return 0;
}
