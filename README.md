# PQC Final Project

This repository is created for the implementation of large number multiplication on Rasberry Pi, including hand-writing algorithms and a testing tool forked from `mkannwischer/aarch64-bench` for benchmarking the performance of multiplicaiton.

## How to benchmark

Run the following commands:

```
cd aarch64-bench
make clean
make CYCLE=PERF
sudo ./bench
```

## Current Implementation

- 2025-05-31: Toom-3
  - Perfomance: ~58000 cycles (384 limbs 16-bit number)
