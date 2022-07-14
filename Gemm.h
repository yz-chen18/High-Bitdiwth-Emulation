#pragma once

#include "common.h"

void Gemm32_FX(int m, int n, int k, const float* host_A, const float* host_B, float* host_C);
void Gemm32(int m, int n, int k, const float* host_A, const float* host_B, float* host_C);
void Gemm16(int m, int n, int k, const Fp16* host_A, const Fp16* host_B, float* host_C);
void Gemm(int m, int n, int k, const Fp16* host_A_Hi, const Fp16* host_A_Lo, const Fp16* host_B_Hi, const Fp16* host_B_Lo, 
  const Fp16* host_C, Fp32* host_D);
void stride_Gemm32_FX(int m, int n, int k, int s, const float* host_A, const float* host_B, float* host_C);