#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>

#include "common.h"
#include "transformer.h"
//#include "cublas_utils.h"

int main(void) {
    cublasHandle_t handle = NULL;
    cublasCreate(&handle);
    int m = 8;
    int k = 128;
    int n = 128;
    int s = 8;

    float* host_A;
    float* host_B;
    float* host_C;

    Fp16* host_A_Hi;
    Fp16* host_A_Lo;
    Fp16* host_B_Hi;
    Fp16* host_B_Lo;

    host_A = (float*)malloc(sizeof(float) * m * k);
    host_B = (float*)malloc(sizeof(float) * n * k);
    host_C = (float*)malloc(sizeof(float) * m * n);

    host_A_Hi = (Fp16*)malloc(sizeof(Fp16) * m * k);
    host_A_Lo = (Fp16*)malloc(sizeof(Fp16) * m * k);
    host_B_Hi = (Fp16*)malloc(sizeof(Fp16) * k * n);
    host_B_Lo = (Fp16*)malloc(sizeof(Fp16) * k * n);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            Fp32 m;
            //m.ui = 0x45177fe6;
            host_A[i*k + j] = i*k + j;
        }
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            Fp32 m;
            //m.ui = 0x3f7a574e;
            host_B[i*n + j] = i*n + j;
        }
    }

    float* dev_A;
    float* dev_B;
    float* dev_C;

    cudaMalloc((void**) &dev_A, m * k * sizeof(float));
    cudaMalloc((void**) &dev_B, n * k * sizeof(float));
    cudaMalloc((void**) &dev_C, m * n * sizeof(float));

    cudaMemcpy(dev_A, host_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, host_B, n * k * sizeof(float), cudaMemcpyHostToDevice);

    const float alpha = 1.0;
    const float beta = 1.0;

    cudaMemset(dev_C, 0, sizeof(float) * m * n);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dev_A, CUDA_R_32F, m, dev_B, CUDA_R_32F, k, &beta, dev_C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    printf("CUDA\n");
    cudaMemcpy(host_C, dev_C, n * m * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%e ", host_C[i*n + j]);
        }
        printf("\n");
    }

    matrix_stride_transpose(k, n, s, host_B);
    cudaMemcpy(dev_B, host_B, n * k * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(dev_C, 0, sizeof(float) * m * n);
    for (int i = 0; i < k / s; i++)
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, s, &alpha, dev_A + i*m, CUDA_R_32F, m*k/s, dev_B + i*s, CUDA_R_32F, k, &beta, dev_C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);    
    //cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, s, &alpha, dev_A + m, CUDA_R_32F, m*2, dev_B + 4, CUDA_R_32F, k, &beta, dev_C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT); 
    //cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, 4, &alpha, dev_A, CUDA_R_32F, m*2, dev_B + 8, CUDA_R_32F, k, &beta, dev_C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT); 
    //cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, 4, &alpha, dev_A, CUDA_R_32F, m*2, dev_B + 12, CUDA_R_32F, k, &beta, dev_C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT); 
    
    printf("TENSOR\n");
    cudaMemcpy(host_C, dev_C, n * m * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%e ", host_C[i*n + j]);
        }
        printf("\n");
    }
    return 0;
}