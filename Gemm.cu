#include <math.h>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "Gemm.h"
#include "transformer.h"
//#include "error.h"
//#include "cublas_utils.h"

__global__ void add( float *a, float *b, float *c , int N) {

    // the initial index that this thread will work on
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    // In this above example code, we assume a linear set of blocks of threads in the 'x' dimension,
    // which is declared in main below when we run this function.

    // The actual computation is being done by individual threads
    // in each of the blocks.
    // e.g. we use 4 blocks and 2 threads per block (8 threads will run in parallel)
    //      and our total array size N is 8
    //      the thread whose threadIdx.x is 0 within block 0 will compute c[0],
    //          because tid = (2 * 0)  + 0
    //      the thread whose threadIdx.x is 0 within block 1 will compute c[2],
    //          because tid = (2 * 1) + 0
    //      the thread whose threadIdx.x is 1 within block 1 will compute c[3],
    //          because tid = (2 * 1) + 1
    //
    //     The following while loop will execute once for this simple example:
    //          c[0] through c[7] will be computed concurrently
    //
    if (tid < N) {
        c[tid] = a[tid] + b[tid];       // The actual computation done by the thread
    }
}

void stride_Gemm32_FX(int m, int n, int k, int s, const float* host_A, const float* host_B, float* host_C) {
  //if ((m % 4 != 0) || (k % 8 != 0)) error("Gemm", "m must be a multiple of 4, k must be a multiple of 8");
  cublasHandle_t handle = NULL;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

  float* dev_A;
  Fp16* dev_A_Hi;
  Fp16* dev_A_Lo;
  Fp16* dev_A_Lofx;

  float* dev_B;
  Fp16* dev_B_Hi;
  Fp16* dev_B_Lo;
  Fp16* dev_B_Lofx;

  float* dev_C;

  cudaMalloc((void**) &dev_A, sizeof(float) * m * k);
  cudaMalloc((void**) &dev_A_Hi, sizeof(Fp16) * m  * k);
  cudaMalloc((void**) &dev_A_Lo, sizeof(Fp16) * m  * k);
  cudaMalloc((void**) &dev_A_Lofx, sizeof(Fp16) * m  * k);

  cudaMalloc((void**) &dev_B, sizeof(float) * n * k);
  cudaMalloc((void**) &dev_B_Hi, sizeof(Fp16) * n  * k);
  cudaMalloc((void**) &dev_B_Lo, sizeof(Fp16) * n  * k);
  cudaMalloc((void**) &dev_B_Lofx, sizeof(Fp16) * n  * k);

  cudaMalloc((void**) &dev_C, sizeof(float) * m * n);

  Fp16* host_A_Hi = (Fp16*)malloc(sizeof(Fp16) * m * k);
  Fp16* host_A_Lo = (Fp16*)malloc(sizeof(Fp16) * m * k);
  Fp16* host_A_Lofx = (Fp16*)malloc(sizeof(Fp16) * m * k);
  Fp16* host_B_Hi = (Fp16*)malloc(sizeof(Fp16) * n * k);
  Fp16* host_B_Lo = (Fp16*)malloc(sizeof(Fp16) * n * k);
  Fp16* host_B_Lofx = (Fp16*)malloc(sizeof(Fp16) * n * k);

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      toFp32_F_FX(host_A[i*k + j], &host_A_Hi[i*k + j], &host_A_Lo[i*k + j], &host_A_Lofx[i*k + j]);
    }
  }

  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n; j++) {
      toFp32_F_FX(host_B[i*n + j], &host_B_Hi[i*n + j], &host_B_Lo[i*n + j], &host_B_Lofx[i*n + j]);
    }
  }

  matrix_stride_transpose(k, n, s, host_B_Hi);
  matrix_stride_transpose(k, n, s, host_B_Lo);
  matrix_stride_transpose(k, n, s, host_B_Lofx);

  cudaMemcpy(dev_A, host_A, sizeof(float) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, host_B, sizeof(float) * n * k, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_C, host_C, sizeof(float) * m * n, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_A_Hi, host_A_Hi, sizeof(Fp16) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_A_Lo, host_A_Lo, sizeof(Fp16) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_A_Lofx, host_A_Lofx, sizeof(Fp16) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B_Hi, host_B_Hi, sizeof(Fp16) * n * k, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B_Lo, host_B_Lo, sizeof(Fp16) * n * k, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B_Lofx, host_B_Lofx, sizeof(Fp16) * n * k, cudaMemcpyHostToDevice);

  float alpha0 = 1.0f;
  float alpha1 = pow(0.5f, 12);
  float zero = 0.0f;
  float beta = 1.0f;

  float* dev_C_partial;
  cudaMalloc((void**) &dev_C_partial, sizeof(float) * m * n);

  for (int i = 0; i < k / s; i++) {
    //todo sum up partial sum with cuda core
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, s, &alpha0, dev_A_Hi + i*m, CUDA_R_16F, m*k/s, dev_B_Hi + i*s, CUDA_R_16F, k, &zero, dev_C_partial, CUDA_R_32F, m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    add<<<(m * n + 31)/32, 32>>>(dev_C_partial, dev_C, dev_C, m*n);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, s, &alpha1, dev_A_Hi + i*m, CUDA_R_16F, m*k/s, dev_B_Lo + i*s, CUDA_R_16F, k, &zero, dev_C_partial, CUDA_R_32F, m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    add<<<(m * n + 31)/32, 32>>>(dev_C_partial, dev_C, dev_C, m*n);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, s, &alpha1, dev_A_Lo + i*m, CUDA_R_16F, m*k/s, dev_B_Hi + i*s, CUDA_R_16F, k, &zero, dev_C_partial, CUDA_R_32F, m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    add<<<(m * n + 31)/32, 32>>>(dev_C_partial, dev_C, dev_C, m*n);
    //cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, s, &alpha1, dev_A_Hi + i*m, CUDA_R_16F, m*k/s, dev_B_Lofx + i*s, CUDA_R_16F, k, &zero, dev_C_partial, CUDA_R_32F, m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    //add<<<(m * n + 31)/32, 32>>>(dev_C_partial, dev_C, dev_C, m*n);
    //cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, s, &alpha1, dev_A_Lofx + i*m, CUDA_R_16F, m*k/s, dev_B_Hi + i*s, CUDA_R_16F, k, &zero, dev_C_partial, CUDA_R_32F, m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    //add<<<(m * n + 31)/32, 32>>>(dev_C_partial, dev_C, dev_C, m*n);
  }
  
  cudaMemcpy(host_C, dev_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

  free(host_A_Hi);
  free(host_A_Lo);
  free(host_A_Lofx);
  free(host_B_Hi);
  free(host_B_Lo);
  free(host_B_Lofx);
  
  cudaFree(dev_A);
  cudaFree(dev_A_Hi);
  cudaFree(dev_A_Lo);
  cudaFree(dev_A_Lofx);
  cudaFree(dev_B);
  cudaFree(dev_B_Hi);
  cudaFree(dev_B_Lo);
  cudaFree(dev_B_Lofx);
  cudaFree(dev_C);
}

void Gemm32_FX(int m, int n, int k, const float* host_A, const float* host_B, float* host_C) {
  //if ((m % 4 != 0) || (k % 8 != 0)) error("Gemm", "m must be a multiple of 4, k must be a multiple of 8");
  cublasHandle_t handle = NULL;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

  float* dev_A;
  Fp16* dev_A_Hi;
  Fp16* dev_A_Lo;
  Fp16* dev_A_Lofx;

  float* dev_B;
  Fp16* dev_B_Hi;
  Fp16* dev_B_Lo;
  Fp16* dev_B_Lofx;

  float* dev_C;

  cudaMalloc((void**) &dev_A, sizeof(float) * m * k);
  cudaMalloc((void**) &dev_A_Hi, sizeof(Fp16) * m  * k);
  cudaMalloc((void**) &dev_A_Lo, sizeof(Fp16) * m  * k);
  cudaMalloc((void**) &dev_A_Lofx, sizeof(Fp16) * m  * k);

  cudaMalloc((void**) &dev_B, sizeof(float) * n * k);
  cudaMalloc((void**) &dev_B_Hi, sizeof(Fp16) * n  * k);
  cudaMalloc((void**) &dev_B_Lo, sizeof(Fp16) * n  * k);
  cudaMalloc((void**) &dev_B_Lofx, sizeof(Fp16) * n  * k);

  cudaMalloc((void**) &dev_C, sizeof(float) * m * n);

  Fp16* host_A_Hi = (Fp16*)malloc(sizeof(Fp16) * m * k);
  Fp16* host_A_Lo = (Fp16*)malloc(sizeof(Fp16) * m * k);
  Fp16* host_A_Lofx = (Fp16*)malloc(sizeof(Fp16) * m * k);
  Fp16* host_B_Hi = (Fp16*)malloc(sizeof(Fp16) * n * k);
  Fp16* host_B_Lo = (Fp16*)malloc(sizeof(Fp16) * n * k);
  Fp16* host_B_Lofx = (Fp16*)malloc(sizeof(Fp16) * n * k);

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      toFp32_F_FX(host_A[i*k + j], &host_A_Hi[i*k + j], &host_A_Lo[i*k + j], &host_A_Lofx[i*k + j]);
    }
  }

  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n; j++) {
      toFp32_F_FX(host_B[i*n + j], &host_B_Hi[i*n + j], &host_B_Lo[i*n + j], &host_B_Lofx[i*n + j]);
    }
  }

  cudaMemcpy(dev_A, host_A, sizeof(float) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, host_B, sizeof(float) * n * k, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_C, host_C, sizeof(float) * m * n, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_A_Hi, host_A_Hi, sizeof(Fp16) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_A_Lo, host_A_Lo, sizeof(Fp16) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_A_Lofx, host_A_Lofx, sizeof(Fp16) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B_Hi, host_B_Hi, sizeof(Fp16) * n * k, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B_Lo, host_B_Lo, sizeof(Fp16) * n * k, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B_Lofx, host_B_Lofx, sizeof(Fp16) * n * k, cudaMemcpyHostToDevice);

  float alpha0 = 1.0f;
  float alpha1 = pow(0.5f, 12);
  float beta = 1.0f;

  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha0, dev_A_Hi, CUDA_R_16F, m, dev_B_Hi, CUDA_R_16F, k, &beta, dev_C, CUDA_R_32F, m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha1, dev_A_Hi, CUDA_R_16F, m, dev_B_Lo, CUDA_R_16F, k, &beta, dev_C, CUDA_R_32F, m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha1, dev_A_Lo, CUDA_R_16F, m, dev_B_Hi, CUDA_R_16F, k, &beta, dev_C, CUDA_R_32F, m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha1, dev_A_Hi, CUDA_R_16F, m, dev_B_Lofx, CUDA_R_16F, k, &beta, dev_C, CUDA_R_32F, m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha1, dev_A_Lofx, CUDA_R_16F, m, dev_B_Hi, CUDA_R_16F, k, &beta, dev_C, CUDA_R_32F, m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  
  cudaMemcpy(host_C, dev_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

  free(host_A_Hi);
  free(host_A_Lo);
  free(host_A_Lofx);
  free(host_B_Hi);
  free(host_B_Lo);
  free(host_B_Lofx);
  
  cudaFree(dev_A);
  cudaFree(dev_A_Hi);
  cudaFree(dev_A_Lo);
  cudaFree(dev_A_Lofx);
  cudaFree(dev_B);
  cudaFree(dev_B_Hi);
  cudaFree(dev_B_Lo);
  cudaFree(dev_B_Lofx);
  cudaFree(dev_C);
}

//todo: template
void Gemm16(int m, int n, int k, const Fp16* host_A, const Fp16* host_B, float* host_C) {
  //if ((m % 4 != 0) || (k % 8 != 0)) error("Gemm", "m must be a multiple of 4, k must be a multiple of 8");
  cublasHandle_t handle = NULL;
  cublasCreate(&handle);
  //cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

  Fp16* dev_A;
  Fp16* dev_B;
  float* dev_C;

  cudaMalloc((void**) &dev_A, sizeof(Fp16) * m * k);
  cudaMalloc((void**) &dev_B, sizeof(Fp16) * n * k);
  cudaMalloc((void**) &dev_C, sizeof(float) * m * n);

  cudaMemcpy(dev_A, host_A, sizeof(Fp16) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, host_B, sizeof(Fp16) * n * k, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_C, host_C, sizeof(float) * m * n, cudaMemcpyHostToDevice);

  float alpha = 1.0f;
  float beta = 0.0f;

  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, (void*)dev_A, CUDA_R_16F, m, (void*)dev_B, CUDA_R_16F, k, &beta, (void*)dev_C, CUDA_R_32F, m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  
  cudaMemcpy(host_C, dev_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

  cudaFree(dev_A);
  cudaFree(dev_B);
  cudaFree(dev_C);
}

void Gemm32(int m, int n, int k, const float* host_A, const float* host_B, float* host_C) {
  //if ((m % 4 != 0) || (k % 8 != 0)) error("Gemm", "m must be a multiple of 4, k must be a multiple of 8");
  cublasHandle_t handle = NULL;
  cublasCreate(&handle);
  //cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

  float* dev_A;
  float* dev_B;
  float* dev_C;

  cudaMalloc((void**) &dev_A, sizeof(float) * m * k);
  cudaMalloc((void**) &dev_B, sizeof(float) * n * k);
  cudaMalloc((void**) &dev_C, sizeof(float) * m * n);

  cudaMemcpy(dev_A, host_A, sizeof(float) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, host_B, sizeof(float) * n * k, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_C, host_C, sizeof(float) * m * n, cudaMemcpyHostToDevice);

  float alpha = 1.0f;
  float beta = 0.0f;

  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, (void*)dev_A, CUDA_R_32F, m, (void*)dev_B, CUDA_R_32F, k, &beta, (void*)dev_C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  
  cudaMemcpy(host_C, dev_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

  cudaFree(dev_A);
  cudaFree(dev_B);
  cudaFree(dev_C);
}

void Gemm(int m, int n, int k, const Fp16* host_A_Hi, const Fp16* host_A_Lo, const Fp16* host_B_Hi, const Fp16* host_B_Lo, 
  const Fp16* host_C, Fp32* host_D) {
    //if ((m % 4 != 0) || (k % 8 != 0)) error("Gemm", "m must be a multiple of 4, k must be a multiple of 8");
    cublasHandle_t handle = NULL;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    Fp16* dev_A_Hi;
    Fp16* dev_A_Lo;
    Fp16* dev_B_Hi;
    Fp16* dev_B_Lo;
    Fp16* dev_C;

    const float alpha = 1.0;
    const float beta = 0.0;

    cudaMalloc((void**) &dev_A_Hi, m * k * sizeof(Fp16));
    cudaMalloc((void**) &dev_A_Lo, m * k * sizeof(Fp16));
    cudaMalloc((void**) &dev_B_Hi, n * k * sizeof(Fp16));
    cudaMalloc((void**) &dev_B_Lo, n * k * sizeof(Fp16));
    cudaMalloc((void**) &dev_C, m * n * sizeof(Fp16));

    cudaMemcpy(dev_A_Hi, host_A_Hi, m * k * sizeof(Fp16), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_A_Lo, host_A_Lo, m * k * sizeof(Fp16), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B_Hi, host_B_Hi, n * k * sizeof(Fp16), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B_Lo, host_B_Lo, n * k * sizeof(Fp16), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_C, host_C, m * n * sizeof(Fp16), cudaMemcpyHostToDevice);

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dev_A_Hi, CUDA_R_16F, m, dev_B_Hi, CUDA_R_16F, k, &beta, dev_C, CUDA_R_16F, m, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}