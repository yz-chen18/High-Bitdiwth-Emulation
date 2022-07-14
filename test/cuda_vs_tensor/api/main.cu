#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>

#include "common.h"
#include "transformer.h"
#include "Gemm.h"
#include "random_sample.h"

int main() {
    int m = 8;
    int k = 8;
    int n = 8;

    half* A = (half*)malloc(sizeof(half) * m * k);
    half* B = (half*)malloc(sizeof(half) * k * n);
    float* A32 = (float*)malloc(sizeof(float) * m * k);
    float* B32 = (float*)malloc(sizeof(float) * k * n);
    float* C32 = (float*)malloc(sizeof(float) * m * n);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            A[i + j*m] = 0.0;
            A32[i + j*m] = 0.0;
        }
    }

    A[0] = 1.0;
    A[m] = pow(0.5, 13);
    A32[0] = 1.0;
    A32[m] = pow(0.5, 13);

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            B[i + j*k] = 0.0;
            B32[i + j*k] = 0.0;
        }
    }
    B[0] = 2.0;
    B[1] = pow(0.5, 10) + pow(0.5, 11);
    B32[0] = 2.0;
    B32[1] = pow(0.5, 10) + pow(0.5, 11);

    half* dev_A;
    half* dev_B;

    float* dev_A32;
    float* dev_B32;
    float* dev_C32;

    cudaMalloc((void**) &dev_A, sizeof(half) * m * k);
    cudaMalloc((void**) &dev_B, sizeof(half) * n * k);
    cudaMalloc((void**) &dev_A32, sizeof(float) * m * k);
    cudaMalloc((void**) &dev_B32, sizeof(float) * n * k);
    cudaMalloc((void**) &dev_C32, sizeof(float) * m * n);

    cudaMemcpy(dev_A, A, sizeof(half) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, sizeof(half) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_A32, A32, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B32, B32, sizeof(float) * m * n, cudaMemcpyHostToDevice);

    cublasHandle_t handle = NULL;
    cublasCreate(&handle);

    const float alpha = 1.0;
    const float beta = 0.0;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dev_A32, m, dev_B32, k, &beta, dev_C32, m);

    cudaMemcpy(C32, dev_C32, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            Fp32 fp32;
            fp32.fp = C32[i + j*m];
            printf("%x ", fp32.ui);
        }

        printf("\n");
    }

    //cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dev_A, CUDA_R_16F, m, dev_B, CUDA_R_16F, k, &beta, dev_C32, CUDA_R_32F, m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaMemcpy(C32, dev_C32, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            Fp32 fp32;
            fp32.fp = C32[i + j*m];
            printf("%x ", fp32.ui);
        }

        printf("\n");
    }
}