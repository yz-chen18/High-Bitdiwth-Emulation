#pragma once

#include <cuda_fp16.h>

#define Fp32_SHIFT 127
#define Fp16_SHIFT 15
#define Fp16_EXP_MAX 15
#define Fp16_EXP_MIN -14

typedef union {
    float fp;
    unsigned int ui;
} Fp32;

typedef union {
    unsigned short us;
    half fp;
} Fp16;

enum RANGE{
    NORMAL = 0,
    POSITIVE = 1,
    ZERO_TO_ONE = 2,
    ONE_TO_ONE = 3,
    ONE_TO_TWO = 4
};