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
    int m = 4;
    int k = 8;
    int n = 1;

    float* host_A;
    float* host_B;
    float* host_C;

    Fp16* host_A_Hi;
    Fp16* host_A_Lo;
    Fp16* host_B_Hi;
    Fp16* host_B_Lo;
    float* host_shift_A;
    float* host_shift_B;

    host_A = (float*)malloc(sizeof(float) * m * k);
    host_B = (float*)malloc(sizeof(float) * n * k);
    host_C = (float*)malloc(sizeof(float) * m * n);

    host_A_Hi = (Fp16*)malloc(sizeof(Fp16) * m * k);
    host_A_Lo = (Fp16*)malloc(sizeof(Fp16) * m * k);
    host_B_Hi = (Fp16*)malloc(sizeof(Fp16) * k * n);
    host_B_Lo = (Fp16*)malloc(sizeof(Fp16) * k * n);

    host_shift_A = (float*)malloc(sizeof(float) * m * k);
    host_shift_B = (float*)malloc(sizeof(float) * n * k);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            Fp32 m;
            m.ui = 0x45177fe6;
            host_A[i*k + j] = m.fp;
            Fp16 hi, lo;
            unsigned int Hi_shift = toFp32_F(m.fp, &hi, &lo);
            host_A_Hi[i*k + j] = hi;
            host_A_Lo[i*k + j] = lo;
            if (Hi_shift >= 0) {
                host_shift_A[i*k + j] = pow(2.f, Hi_shift);
            } else {
                host_shift_A[i*k + j] = pow(0.5f, -Hi_shift);
            }
            
        }
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            Fp32 m;
            m.ui = 0x3f7a574e;
            host_B[i*n + j] = m.fp;
            Fp16 hi, lo;
            unsigned int Hi_shift = toFp32_F(m.fp, &hi, &lo);
            host_B_Hi[i*k + j] = hi;
            host_B_Lo[i*k + j] = lo;
            if (Hi_shift >= 0) {
                host_shift_B[i*k + j] = pow(2.f, Hi_shift);
            } else {
                host_shift_B[i*k + j] = pow(0.5f, -Hi_shift);
            }
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
    const float beta = 0.0;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dev_A, m, dev_B, k, &beta, dev_C, m);

    cudaMemcpy(host_C, dev_C, n * m * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", host_C[i*n + j]);
        }
        printf("\n");
    }

    cudaMemset(dev_C, 0.f, m * n);

    fp16* dev_A_Hi;
    fp16* dev_A_Lo;
    fp16* dev_B_Hi;
    fp16* dev_B_Lo;

    float* dev_shift_A;
    float* dev_shift_B;

    cudaMalloc((void**)&dev_A_Hi, sizeof(fp16) * m * k);
    cudaMalloc((void**)&dev_A_Lo, sizeof(fp16) * m * k);
    cudaMalloc((void**)&dev_B_Hi, sizeof(fp16) * n * k);
    cudaMalloc((void**)&dev_B_Lo, sizeof(fp16) * n * k);
    cudaMalloc((void**)&dev_shift_A, sizeof(float) * m * k);
    cudaMalloc((void**)&dev_shift_B, sizeof(float) * n * k);

    cudaMemcpy(&dev_A_Hi, host_A_Hi, sizeof(fp16) * m * k, cudaMemcpyDeviceToHost);
    cudaMemcpy(&dev_A_Lo, host_A_Lo, sizeof(fp16) * m * k, cudaMemcpyDeviceToHost);
    cudaMemcpy(&dev_B_Hi, host_B_Hi, sizeof(fp16) * n * k, cudaMemcpyDeviceToHost);
    cudaMemcpy(&dev_B_Lo, host_B_Lo, sizeof(fp16) * n * k, cudaMemcpyDeviceToHost);

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dev_A_Hi, CUDA_R_16F, m, dev_B_Hi, CUDA_R_16F, k, &beta, dev_C, CUDA_R_32F, m, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dev_A_Hi, CUDA_R_16F, m, dev_B_Lo, CUDA_R_16F, k, &beta, dev_C, CUDA_R_32F, m, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dev_A_Lo, CUDA_R_16F, m, dev_B_Hi, CUDA_R_16F, k, &beta, dev_C, CUDA_R_32F, m, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    //print_matrix(m, n, host_C, n);
    //cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, (void*)dev_A_Hi, (void*)dev_A_Hi, CUDA_R_16F, 8, (void*)dev_B_Hi, CUDA_R_16F, 8, (void*)dev_B_Hi, (void*)dev_C, CUDA_R_16F, 8, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return 0;
}