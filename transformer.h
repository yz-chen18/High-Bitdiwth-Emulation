#pragma once

#include "common.h"

#define Sign(M) ((M & 0x80000000) >> 31)
#define Exponent(M) ((M & 0x7f800000) >> 23)
#define Significand(M) ((M & 0x007fffff))

int parseFp32(float M, unsigned int* sign, unsigned int* HiE, unsigned int* HiS, unsigned int* R, unsigned int* LoE, unsigned int* LoS, unsigned int* shift);
int parseFp32_FX(float M, unsigned int* sign, unsigned int* HiE, unsigned int* HiS, unsigned int* R, unsigned int* E, unsigned int* LoE, unsigned int* LoS, unsigned int* shift);
int toFp32_F_FX(float M, Fp16* HiM, Fp16* LoM, Fp16* LoFx);
int toFp32_F(float M, Fp16* HiM, Fp16* LoM);
int toFp32(float* M, float Hi_M, float Lo_M);
Fp32 Fp16_To_Fp32(Fp16 m);