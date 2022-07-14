#include <cstdio>
#include <cstring>
#include <cstdlib>

#include "transformer.h"
#include "common.h"
#include "error.h"


unsigned int getSign(float M) {
    Fp32 fp32{M};
    return Sign(fp32.ui);
}

unsigned int getHiE(float M) {
    Fp32 fp32{M};
    #ifdef debug
    printf("%u %u\n", fp32.ui, fp32.ui & 0x0f800000);
    #endif
    return ((fp32.ui & 0x7f800000) >> 23);
}

Fp32 Fp16_To_Fp32(Fp16 m) {
    /*
    if (m.us == 0) return Fp32{0};
    unsigned int E = ((m.us & 0x7c00) >> 10) + Fp32_SHIFT - Fp16_SHIFT;
    unsigned int sign = (m.us & 0x8000) >> 15;
    unsigned int significand = (m.us & 0x03ff);*/

    Fp32 fp32{0.f};
    /*
    fp32.ui |= (E << 23);
    fp32.ui |= (significand << 13);
    fp32.ui |= (sign << 31);*/
    fp32.fp = m.fp;

    return fp32;
}

int parseFp32(float M, unsigned int* sign, unsigned int* HiE, unsigned int* HiS, unsigned int* R, unsigned int* LoE, unsigned int* LoS, unsigned int* shift) {
    Fp32 fp32{M};
    unsigned int t_sign = getSign(M);
    unsigned int t_HiE = getHiE(M) - Fp32_SHIFT + Fp16_SHIFT;
    if ((t_HiE > getHiE(M)) | (t_HiE > Fp16_EXP_MAX + Fp16_SHIFT)) error("parseFp32", "Input Float32 With an Invalid Exponent");
    unsigned int t_HiS = ((fp32.ui & 0x007fe000) >> 13);
    unsigned int t_R = ((fp32.ui & 0x00001000) >> 12);
    unsigned int t_LoS, t_LoE;
    while (t_R) {
        if (((fp32.ui & 0x00000fff) == 0) && ((fp32.ui & 0x00002000) == 0)) {
            break;
        }
        t_HiS += 1;
        fp32.ui = (~fp32.ui + 1) & (0x00001fff);
        break;
    }

    unsigned int shiftE = 12;
    unsigned int i_E = 0x00000800;
    while (i_E > 0) {
        if ((unsigned int)(i_E & fp32.ui) > 0) break;
        shiftE += 1;
        i_E = i_E >> 1;

        #ifdef debug
        printf("%x\n", i_E);
        #endif
    }

    if (i_E != 0) {
        while (i_E > 0x00000400) {
        i_E = i_E >> 1;
        fp32.ui = fp32.ui >> 1;
    }
        while (i_E < 0x00000400) {
            i_E = i_E << 1;
            fp32.ui = fp32.ui << 1;
        }
        t_LoS = (fp32.ui & 0x000003ff);
        t_LoE = t_HiE; // for now, we store the shift in another variable, incase that LoE would underflow
    } else {
        t_LoS = 0;
        t_LoE = 0;
    }

    *sign = t_sign;
    *HiE = t_HiE;
    *HiS = t_HiS;
    *R = t_R;
    *LoE = t_LoE;
    *LoS = t_LoS;
    *shift = shiftE;

    return 0;
}

int parseFp32_FX(float M, unsigned int* sign, unsigned int* HiE, unsigned int* HiS, unsigned int* R, unsigned int* E, unsigned int* LoE, unsigned int* LoS, unsigned int* shift) {
    Fp32 fp32{M};
    unsigned int t_sign = getSign(M);
    unsigned int t_HiE = getHiE(M) - Fp32_SHIFT + Fp16_SHIFT;
    if ((t_HiE > getHiE(M)) | (t_HiE > Fp16_EXP_MAX + Fp16_SHIFT)) error("parseFp32", "Input Float32 With an Invalid Exponent");
    unsigned int t_HiS = ((fp32.ui & 0x007fe000) >> 13);
    unsigned int t_R = ((fp32.ui & 0x00001000) >> 12);
    unsigned int t_E = (fp32.ui & 0x00000800) >> 11;
    unsigned int t_LoS, t_LoE = t_HiE;

    while (t_R) {
        if (((fp32.ui & 0x00000fff) == 0) && ((fp32.ui & 0x00002000) == 0)) {
            t_R = 0; // R represent whether the LoS was the inverse of origin S
            break;
        }
        t_HiS += 1;
        fp32.ui = (~fp32.ui + 1) & (0x00001fff);
        t_E = (fp32.ui & 0x00000800) >> 11;
        t_R = 1;

        // when original HiS == 0x7ff, and R == 1
        if ((t_HiS & 0x00000400) != 0) {
            t_HiS = 0;
            t_HiE += 1;
        }

        #ifdef debug
        printf("t_E: %x\n", t_E);
        #endif
        break;
    }

    unsigned int i_None_Zero = 0x00000800;
    while (i_None_Zero > 0) {
        if ((unsigned int)(i_None_Zero & fp32.ui) > 0) break;
        i_None_Zero = i_None_Zero >> 1;

        #ifdef debug
        printf("%x\n", i_None_Zero);
        #endif
    }

    if (i_None_Zero != 0) {
        t_LoS = (fp32.ui & 0x000007fe) >> 1;
    } else {
        t_LoS = 0;
        t_LoE = 0;
    }

    *sign = t_sign;
    *HiE = t_HiE;
    *HiS = t_HiS;
    *E = t_E;
    *R = t_R;
    *LoE = t_LoE;
    *LoS = t_LoS;
    *shift = 12; // fixed shifting

    return i_None_Zero;
}

// generate the high & low significand of fp32_f from a fp32
int toFp32_F(float M, Fp16* HiM, Fp16* LoM) {
    Fp32 fp32{M};
    unsigned int sign, HiE, HiS, R, LoE, LoS, Lo_shift;
    parseFp32(M, &sign, &HiE, &HiS, &R, &LoE, &LoS, &Lo_shift);
    Fp16 t_HiM{0}, t_LoM{0};

    t_HiM.us |= (sign << 15);
    // set the exponent of Hi_M to be FP16_EXP_MAX to guarantee the exponent of Lo_M would not undeflow FP16_EXP_MIN
    t_HiM.us |= (Fp16_EXP_MAX << 10);
    t_HiM.us |= (HiS);
    t_LoM.us |= (R << 15);
    t_LoM.us |= ((Fp16_EXP_MAX - Lo_shift) << 10);
    t_LoM.us |= (LoS);

    *HiM = t_HiM;
    *LoM = t_LoM;

    // return the shift between Fp32_F and Fp32
    return HiE - Fp16_EXP_MAX;
}

// generate the high & low significand of fp32_f from a fp32 with fixed E, LoFx is to represent correct Lo considering different E
int toFp32_F_FX(float M, Fp16* HiM, Fp16* LoM, Fp16* LoFx) {
    half h = M;
    HiM->fp = h;
    LoM->fp = (M - Fp16_To_Fp32(*HiM).fp) * 4096;
    LoFx->fp = 0.0f;

   /*
    Fp32 fp32{M};
    unsigned int sign, HiE, HiS, R, E, LoE, LoS, Lo_shift;
    parseFp32_FX(M, &sign, &HiE, &HiS, &R, &E, &LoE, &LoS, &Lo_shift);
    Fp16 t_HiM{0}, t_LoM{0}, t_LoFx{0};

    t_HiM.us |= (sign << 15);
    // set the exponent of Hi_M to be FP16_EXP_MAX to guarantee the exponent of Lo_M would not undeflow FP16_EXP_MIN
    t_HiM.us |= (HiE << 10);
    t_HiM.us |= (HiS);
    t_LoM.us |= ((R^sign) << 15);
    t_LoM.us |= (LoE << 10);
    t_LoM.us |= (LoS);
    
    if (E == 0) {
        t_LoFx.us |= ((1-(R^sign)) << 15);
        t_LoFx.us |= (LoE << 10);
    }

    *HiM = t_HiM;
    *LoM = t_LoM;
    *LoFx = t_LoFx;*/

    return 0;
}

unsigned int toFp32(float Hi_M, float Lo_M, float Lo_shift) {
    Fp32 fp32{0.f};
    Fp32 fp_Hi{Hi_M};
    Fp32 fp_Lo{Lo_M};

    unsigned int Hi_E = getHiE(Hi_M) - Fp16_SHIFT + Fp32_SHIFT;
    unsigned int Lo_E = getHiE(Lo_M) - Fp16_SHIFT + Fp32_SHIFT;

    return fp32.ui;
}

int matrix_stride_transpose(int m, int n, int s, float* fp) {
    if (m % s != 0) error("matrix_stride_transpose", "m must be a multiplicate of s");
    float* t = (float*)malloc(sizeof(float) * m * n);
    int n_block = m / s;
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int index = i*n + j;
            t[(index / m) * m + (index % n_block) * s + (index % m) / n_block] = fp[i*n + j];
        }
    }

    memcpy(fp, t, sizeof(float) * m * n);
    free(t);

    return 0;

}

int matrix_stride_transpose(int m, int n, int s, Fp16* fp) {
    if (m % s != 0) error("matrix_stride_transpose", "m must be a multiplicate of s");
    Fp16* t = (Fp16*)malloc(sizeof(Fp16) * m * n);
    int n_block = m / s;
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int index = i*n + j;
            t[(index / m) * m + (index % n_block) * s + (index % m) / n_block] = fp[i*n + j];
        }
    }

    memcpy(fp, t, sizeof(Fp16) * m * n);
    free(t);

    return 0;

}