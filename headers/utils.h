# ifndef UTILS_H
# define UTILS_H

#include <stdint.h>

// generate a random number in [0, P)
uint64_t rand64_modp(void);

// fast modular multiplication
uint64_t modmul(uint64_t a, uint64_t b, uint64_t p);

// fast modular exponentiation
uint64_t modpow(uint64_t base, uint64_t exp, uint64_t p);

// modular addition
uint64_t modadd(uint64_t a, uint64_t b, uint64_t p);

// modular subtraction
uint64_t modsub(uint64_t a, uint64_t b, uint64_t p);

// modular inverse using Fermat's little theorem
uint64_t modinv(uint64_t x, uint64_t p);
# endif // UTILS_H