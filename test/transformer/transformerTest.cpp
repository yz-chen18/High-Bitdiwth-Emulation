#include <cstdio>
#include <cmath>
#include "transformer.h"
#include "random_sample.h"
#include "common.h"

int toFp32_F_TEST() {
    unsigned int sign_A, HiE_A, HiS_A, R_A, LoE_A, LoS_A, shift_A;
    Fp32 fp32_A;
    fp32_A.fp = /*0x44d20a18*/rand_sample(NORMAL);
    //fp32_A.ui = 0x0AD070C6;
    parseFp32(fp32_A.fp, &sign_A, &HiE_A, &HiS_A, &R_A, &LoE_A, &LoS_A, &shift_A);

    printf("M: %e %x\n", fp32_A.fp, fp32_A.ui);
    Fp16 HiM, LoM;
    int Hi_shift = toFp32_F(fp32_A.fp, &HiM, &LoM);

    Fp32 LoM32 = Fp16_To_Fp32(LoM);
    Fp32 HiM32 = Fp16_To_Fp32(HiM);
    Fp32 sum{HiM32.fp + LoM32.fp};
    sum.fp *= pow(2.f, Fp32_SHIFT - Fp16_SHIFT);

    if (Hi_shift >= 0) {
        sum.fp *= pow(2.f, Hi_shift);
    } else {
        sum.fp *= pow(0.5f, -Hi_shift);
    }

    printf("R_A %u shift_A %u\n", R_A, shift_A);
    printf("HiM: %e %x\n", HiM32.fp, HiM32.ui);
    printf("LoM: %e %x\n", LoM32.fp, LoM32.ui);
    printf("HiM+LoM: %e %x\n", sum.fp, sum.ui);
}

int toFp32_F_FX_TEST() {
    //::testing::InitGoogleTest(&argc, argv);
    //return RUN_ALL_TESTS();
    
    Fp32 fp32_A;
    fp32_A.ui = 0x38f803ff;
    //fp32_A.ui = 0x43000001;
    //fp32_A.fp = rand_sample(NORMAL);

    printf("M: %e %x\n", fp32_A.fp, fp32_A.ui);
    Fp16 HiM, LoM, LoFx;
    toFp32_F_FX(fp32_A.fp, &HiM, &LoM, &LoFx);
    printf("LoM16: %x\n", LoM.us);

    Fp32 LoM32 = Fp16_To_Fp32(LoM);
    Fp32 HiM32 = Fp16_To_Fp32(HiM);
    Fp32 LoFx32 = Fp16_To_Fp32(LoFx);
    Fp32 sum{HiM32.fp + LoM32.fp * pow(0.5f, 12) + LoFx32.fp * pow(0.5f, 12)};

    printf("HiM: %e %x\n", HiM32.fp, HiM32.ui);
    printf("LoM: %e %x\n", LoM32.fp, LoM32.ui);
    printf("LoFx: %e %x\n", LoFx32.fp, LoFx32.ui);
    printf("HiM+LoM+LoFx: %e %x\n", sum.fp, sum.ui);
    printf("variance: %e\n", abs(fp32_A.fp - sum.fp) / fp32_A.fp);
    return 0;
}

int MUL_FX_TEST() {
    //::testing::InitGoogleTest(&argc, argv);
    //return RUN_ALL_TESTS();
    
    Fp32 fp32_A, fp32_B;
    //fp32_A.ui = 0x3cf90337;
    fp32_A.fp = rand_sample(NORMAL);
    fp32_B.fp = rand_sample(NORMAL);

    printf("A: %e %x\n", fp32_A.fp, fp32_A.ui);
    printf("B: %e %x\n", fp32_B.fp, fp32_B.ui);
    Fp16 HiM_A, LoM_A, LoFx_A;
    Fp16 HiM_B, LoM_B, LoFx_B;
    toFp32_F_FX(fp32_A.fp, &HiM_A, &LoM_A, &LoFx_A);
    toFp32_F_FX(fp32_B.fp, &HiM_B, &LoM_B, &LoFx_B);

    Fp32 LoM32_A = Fp16_To_Fp32(LoM_A);
    Fp32 HiM32_A = Fp16_To_Fp32(HiM_A);
    Fp32 LoFx32_A = Fp16_To_Fp32(LoFx_A);
    Fp32 LoM32_B = Fp16_To_Fp32(LoM_B);
    Fp32 HiM32_B = Fp16_To_Fp32(HiM_B);
    Fp32 LoFx32_B = Fp16_To_Fp32(LoFx_B);


    Fp32 fp32_C, fp32_D;
    //fp32_A.ui = 0x3cf90337;
    fp32_C.fp = rand_sample(NORMAL);
    fp32_D.fp = rand_sample(NORMAL);

    printf("C: %e %x\n", fp32_C.fp, fp32_C.ui);
    printf("D: %e %x\n", fp32_D.fp, fp32_D.ui);
    Fp16 HiM_C, LoM_C, LoFx_C;
    Fp16 HiM_D, LoM_D, LoFx_D;
    toFp32_F_FX(fp32_C.fp, &HiM_C, &LoM_C, &LoFx_C);
    toFp32_F_FX(fp32_D.fp, &HiM_D, &LoM_D, &LoFx_D);

    Fp32 LoM32_C = Fp16_To_Fp32(LoM_C);
    Fp32 HiM32_C = Fp16_To_Fp32(HiM_C);
    Fp32 LoFx32_C = Fp16_To_Fp32(LoFx_C);
    Fp32 LoM32_D = Fp16_To_Fp32(LoM_D);
    Fp32 HiM32_D = Fp16_To_Fp32(HiM_D);
    Fp32 LoFx32_D = Fp16_To_Fp32(LoFx_D);

    float alpha = pow(0.5f, 12);

    Fp32 sum1{HiM32_A.fp * HiM32_C.fp + HiM32_B.fp * HiM32_D.fp};
    Fp32 sum2{HiM32_A.fp * LoM32_C.fp + HiM32_B.fp * LoM32_D.fp};
    Fp32 sum3{HiM32_A.fp * LoFx32_C.fp + HiM32_B.fp * LoFx32_D.fp};
    Fp32 sum4{LoM32_A.fp * HiM32_C.fp + LoM32_B.fp * HiM32_D.fp};
    Fp32 sum5{LoFx32_A.fp * HiM32_C.fp + LoFx32_B.fp * HiM32_D.fp};

    Fp32 e_sum{sum1.fp + sum2.fp * alpha + sum3.fp * alpha + sum4.fp * alpha + sum5.fp * alpha};
    Fp32 sum{fp32_A.fp * fp32_C.fp + fp32_B.fp * fp32_D.fp};
    printf("A*C + B*D: %e %x\n", sum.fp, sum.ui);
    printf("Emulated A*C + B*D: %e %x\n", e_sum.fp, e_sum.ui);

    Fp32 e_A{HiM32_A.fp + LoM32_A.fp * pow(0.5f, 12) + LoFx32_A.fp * pow(0.5f, 12)};
    Fp32 e_B{HiM32_B.fp + LoM32_B.fp * pow(0.5f, 12) + LoFx32_B.fp * pow(0.5f, 12)};
    Fp32 e_C{HiM32_C.fp + LoM32_C.fp * pow(0.5f, 12) + LoFx32_C.fp * pow(0.5f, 12)};
    Fp32 e_D{HiM32_D.fp + LoM32_D.fp * pow(0.5f, 12) + LoFx32_D.fp * pow(0.5f, 12)};

    printf("e_A: %e %x\n", e_A.fp, e_A.ui);
    printf("e_B: %e %x\n", e_B.fp, e_B.ui);
    printf("e_C: %e %x\n", e_C.fp, e_C.ui);
    printf("e_D: %e %x\n", e_D.fp, e_D.ui);
    e_sum.fp = e_A.fp*e_C.fp + e_B.fp*e_D.fp;
    printf("e_A*e_C + e_B*e_D: %e %x\n", e_sum.fp, e_sum.ui);
    return 0;
}

void TRANSPOSE_TEST() {
    int m = 32;
    int n = 8;

    float* A = (float*)malloc(sizeof(float) * m * n);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A[i*n + j] = i*n + j;
        }
    }

    matrix_stride_transpose(m, n, 8, A);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", A[i*n + j]);
        }
        printf("\n");
    }
}

typedef union {
    double db;
    unsigned long ul;
} Fp64;

void UPGRADE_TEST() {
    Fp32 fp32;
    fp32.fp = rand_sample(NORMAL);
    Fp64 fp64;
    fp64.db = fp32.fp;
    printf("fp32: %e %p, fp64: %e %p\n", fp32.fp, fp32.ui, fp64.db, fp64.ul);
}

int main(int argc, char* argv[]) {
    //::testing::InitGoogleTest(&argc, argv);
    //return RUN_ALL_TESTS();
    
    printf("%e %e %e %e\n", rand_sample(NORMAL), rand_sample(ZERO_TO_ONE), rand_sample(ONE_TO_ONE), rand_sample(ONE_TO_TWO));
    return 0;
}