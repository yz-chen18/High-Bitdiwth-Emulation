#pragma once

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
} Fp16;