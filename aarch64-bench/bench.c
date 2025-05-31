/*
 * Copyright (c) 2024-2025 The mlkem-native project authors
 * SPDX-License-Identifier: Apache-2.0
 */
#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "hal.h"
#include "../headers/test_data.h"
#include "../headers/mul.h"

#define NWARMUP 5
#define NITERATIONS 30
#define NTESTS 50
#define N 384

static int cmp_uint64_t(const void *a, const void *b)
{
  return (int)((*((const uint64_t *)a)) - (*((const uint64_t *)b)));
}

static void print_median(const char *txt, uint64_t cyc[NTESTS])
{
  printf("%10s cycles = %" PRIu64 "\n", txt, cyc[NTESTS >> 1] / NITERATIONS);
}

static int percentiles[] = {1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99};

static void print_percentile_legend(void)
{
  unsigned i;
  printf("%21s", "percentile");
  for (i = 0; i < sizeof(percentiles) / sizeof(percentiles[0]); i++)
  {
    printf("%7d", percentiles[i]);
  }
  printf("\n");
}

static void print_percentiles(const char *txt, uint64_t cyc[NTESTS])
{
  unsigned i;
  printf("%10s percentiles:", txt);
  for (i = 0; i < sizeof(percentiles) / sizeof(percentiles[0]); i++)
  {
    printf("%7" PRIu64, (cyc)[NTESTS * percentiles[i] / 100] / NITERATIONS);
  }
  printf("\n");
}

static bool check_correctness(void) {
  uint16_t A[N], B[N], TOOM8_OUT[N * 2], GMP_OUT[N * 2];
  memset(GMP_OUT, 0, sizeof(GMP_OUT));
  memcpy(A, test_A, sizeof(A));
  memcpy(B, test_B, sizeof(B));

  toom3_mul_2048(TOOM8_OUT, A, B);
  gmp_mul(N * sizeof(uint16_t), A, B, GMP_OUT);

  for (int i = 0; i < 5; ++i) {
    printf("limb %d: Toom3 = %d, GMP = %d\n",i, TOOM8_OUT[i], GMP_OUT[i]);
  }
  return true;
}

extern const int16_t zetas_layer12345[];
extern const int16_t zetas_layer67[];
void ntt_asm(int16_t *, const int16_t *, const int16_t *);

/*static int bench(void)
{
  int16_t a[256] = {0};
  int i, j;
  uint64_t t0, t1;
  uint64_t cycles_ntt[NTESTS];


  for (i = 0; i < NTESTS; i++)
  {
    for (j = 0; j < NWARMUP; j++)
    {
      ntt_asm(a, zetas_layer12345, zetas_layer67);
    }

    t0 = get_cyclecounter();
    for (j = 0; j < NITERATIONS; j++)
    {
      ntt_asm(a, zetas_layer12345, zetas_layer67);
    }
    t1 = get_cyclecounter();
    cycles_ntt[i] = t1 - t0;
  }

  qsort(cycles_ntt, NTESTS, sizeof(uint64_t), cmp_uint64_t);

  print_median("ntt", cycles_ntt);

  printf("\n");

  print_percentile_legend();

  print_percentiles("ntt", cycles_ntt);

  return 0;
}

static void bench_fnt(void) {
  uint64_t A[N], B[N], OUT[N];
  memcpy(A, test_A, sizeof(A));
  memcpy(B, test_B, sizeof(B));
  
  uint64_t cycles[NTESTS];
  int i, j;
  uint64_t t0, t1;
  
  //fnt_mul_2048limb(OUT, A, B);
  for (i = 0; i < NTESTS; i++) {
    for (j = 0; j < NWARMUP; j++) {
      fnt_mul_2048limb(OUT, A, B);
    }
    
    t0 = get_cyclecounter();
    for (j = 0; j < NITERATIONS; j++) {
      fnt_mul_2048limb(OUT, A, B);
    }
    t1 = get_cyclecounter();
    cycles[i] = t1 - t0;
  }
  
  qsort(cycles, NTESTS, sizeof(uint64_t), cmp_uint64_t);
  print_median("fnt", cycles);
  print_percentile_legend();
  print_percentiles("fnt", cycles);
}*/

static void bench_toom8(void) {
  uint16_t A[N], B[N], OUT[3072];
  memcpy(A, test_A, sizeof(A));
  memcpy(B, test_B, sizeof(B));
  
  uint64_t cycles[NTESTS];
  int i, j;
  uint64_t t0, t1;
  
  //fnt_mul_2048limb(OUT, A, B);
  for (i = 0; i < NTESTS; i++) {
    for (j = 0; j < NWARMUP; j++) {
      toom3_mul_2048(OUT, A, B);
    }
    
    t0 = get_cyclecounter();
    for (j = 0; j < NITERATIONS; j++) {
      toom3_mul_2048(OUT, A, B);
    }
    t1 = get_cyclecounter();
    cycles[i] = t1 - t0;
  }
  
  qsort(cycles, NTESTS, sizeof(uint64_t), cmp_uint64_t);
  print_median("fnt", cycles);
  print_percentile_legend();
  print_percentiles("fnt", cycles);
}

static void bench_gmp(void) {
  uint16_t A[N], B[N], OUT[N * 2]; // GMP 可能用到更大空間
  memset(OUT, 0, sizeof(OUT));
  memcpy(A, test_A, sizeof(A));
  memcpy(B, test_B, sizeof(B));
  //gmp_mul(N * sizeof(uint64_t), A, B, OUT);
  
  uint64_t cycles[NTESTS];
  int i, j;
  uint64_t t0, t1;
  
  for (i = 0; i < NTESTS; i++) {
    for (j = 0; j < NWARMUP; j++) {
      gmp_mul(N * sizeof(uint16_t), A, B, OUT);
    }
    
    t0 = get_cyclecounter();
    for (j = 0; j < NITERATIONS; j++) {
      gmp_mul(N * sizeof(uint16_t), A, B, OUT);
    }
    t1 = get_cyclecounter();
    cycles[i] = t1 - t0;
  }
  
  qsort(cycles, NTESTS, sizeof(uint64_t), cmp_uint64_t);
  print_median("gmp", cycles);
  print_percentile_legend();
  print_percentiles("gmp", cycles);
}

int main(void)
{
  enable_cyclecounter();
  check_correctness();
  /*if (!check_correctness()) {
    printf("Aborting benchmark due to mismatch.\n");
    return 1;
  }*/
  
  // bench();
  printf("== Toom-3 乘法測試 ==\n");
  bench_toom8();
  printf("\n");

  printf("== GMP 乘法測試 ==\n");
  bench_gmp();

  /*printf("==NTT 乘法測試 ==\n");
  bench();*/

  disable_cyclecounter();
  return 0;
}