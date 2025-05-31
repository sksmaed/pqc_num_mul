// modmul_asm.S
.global modmul_asm
.type modmul_asm, %function

// Input:
//   x0 = a
//   x1 = b
//   x2 = modulus p (P = 2^64 - 2^32 + 1)
// Output:
//   x0 = (a * b * R^-1) mod p
modmul_asm:
    // t = a * b = [x4:x3]
    mul     x3, x0, x1          // t_low  = a * b (low 64 bits)
    umulh   x4, x0, x1          // t_high = a * b (high 64 bits)

    // N' = -p^{-1} mod R = 0xffffffff00000001
    mov     x5, 0xffffffff00000001  // N'

    // m = (t_low * N') mod R
    mul     x6, x3, x5

    // m * p = [x8:x7]
    mul     x7, x6, x2          // m * p low
    umulh   x8, x6, x2          // m * p high

    // (t + m * p) >> 64
    adds    x9, x3, x7          // x9 = t_low + m*p_low, sets flags
    adc     x10, x4, x8         // x10 = t_high + m*p_high + carry

    // Conditional subtraction if x10 ≥ p
    subs    xzr, x10, x2        // set flags for comparison (x10 - p)
    csel    x0, x10, x10, lo    // x0 = x10 if x10 < p (leave it)
    csel    x0, x10, x10, hs    // x0 = x10 - p if x10 ≥ p
    sub     x0, x0, x2          // performs x10 - p if hs

    ret
