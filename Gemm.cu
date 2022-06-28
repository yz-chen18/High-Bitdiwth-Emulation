#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "common.h"
#include "error.h"
#include "cublas_utils.h"

void Gemm(int m, int n, int k, const Fp16* host_A_Hi, const Fp16* host_A_Lo, const Fp16* host_B_Hi, const Fp16* host_B_Lo, 
  const Fp16* host_C, Fp32* host_D) {
    if ((m % 4 != 0) || (k % 8 != 0)) error("Gemm", "m must be a multiple of 4, k must be a multiple of 8");
    cublasHandle_t handle = NULL;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

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

    cudaMemcpy(dev_A_Hi, host_A_Hi, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_A_Lo, host_A_Lo, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B_Hi, host_B_Hi, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B_Lo, host_B_Lo, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_C, host_C, cudaMemcpyHostToDevice);

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dev_A_Hi, CUDA_R_16F, m, dev_B_Hi, CUDA_R_16F, k, &beta, dev_C, CUDA_R_16F, m, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}