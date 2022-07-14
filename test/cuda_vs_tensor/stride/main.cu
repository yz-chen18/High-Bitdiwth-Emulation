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
//#include "cublas_utils.h"

#define MAX(m, n) ((m > n) ? m : n)

//#define debug

void find_input(FILE* fp, float* A, float* B, float* C, double* ref_C, int m, int n, int k, int index, double error) {
    int row = index % m;
    int col = index / m;

    float t = 0.0f;
    for (int i = 0; i < k; i++) {
        t += A[i*m + row]*B[col*k + i];
    }
    fprintf(fp, "ref_C: %f, C: %f, t: %f\n", ref_C[index], C[index], t);

    
    for (int i = 0; i < k; i++) {
        fprintf(fp, "%f ", A[i*m + row]);
    }
    fprintf(fp, "\n");

    for (int i = 0; i < k; i++) {
        fprintf(fp, "%f ", B[col*k + i]);
    }
    fprintf(fp, "\n");
    fprintf(fp, "%.2e\n", error);
}

void TEST(int m, int k, int n, RANGE range) {
    cublasHandle_t handle = NULL;
    cublasCreate(&handle);
    
    FILE *fp = fopen("data", "a");

    #ifdef debug
    FILE* fp_debug = fopen("debug", "a");
    #endif

    float* host_A;
    float* host_B;
    float* host_C;

    double* host_A_Fp64;
    double* host_B_Fp64;
    double* host_C_Fp64;

    float* host_C_Fp32;
    float* host_C_Tensor_Stride;
    float* host_C_Tensor;
    float* host_C_CUDA;

    Fp16* host_A_Hi;
    Fp16* host_A_Lo;
    Fp16* host_A_LoFx;
    Fp16* host_B_Hi;
    Fp16* host_B_Lo;
    Fp16* host_B_LoFx;

    float* host_A_Hi32;
    float* host_A_Lo32;
    float* host_A_LoFx32;
    float* host_B_Hi32;
    float* host_B_Lo32;
    float* host_B_LoFx32;

    host_A = (float*)malloc(sizeof(float) * m * k);
    host_B = (float*)malloc(sizeof(float) * n * k);
    host_C = (float*)malloc(sizeof(float) * m * n);

    host_A_Fp64 = (double*)malloc(sizeof(double) * m * k);
    host_B_Fp64 = (double*)malloc(sizeof(double) * n * k);
    host_C_Fp64 = (double*)malloc(sizeof(double) * m * n);    

    host_C_Fp32 = (float*)malloc(sizeof(float) * m * n);
    host_C_CUDA = (float*)malloc(sizeof(float) * m * n);
    host_C_Tensor = (float*)malloc(sizeof(float) * m * n);
    host_C_Tensor_Stride = (float*)malloc(sizeof(float) * m * n);


    host_A_Hi = (Fp16*)malloc(sizeof(Fp16) * m * k);
    host_A_Lo = (Fp16*)malloc(sizeof(Fp16) * m * k);
    host_A_LoFx = (Fp16*)malloc(sizeof(Fp16) * m * k);
    host_B_Hi = (Fp16*)malloc(sizeof(Fp16) * k * n);
    host_B_Lo = (Fp16*)malloc(sizeof(Fp16) * k * n);
    host_B_LoFx = (Fp16*)malloc(sizeof(Fp16) * k * n);

    host_A_Hi32 = (float*)malloc(sizeof(float) * m * k);
    host_A_Lo32 = (float*)malloc(sizeof(float) * m * k);
    host_A_LoFx32 = (float*)malloc(sizeof(float) * m * k);
    host_B_Hi32 = (float*)malloc(sizeof(float) * k * n);
    host_B_Lo32 = (float*)malloc(sizeof(float) * k * n);
    host_B_LoFx32 = (float*)malloc(sizeof(float) * k * n);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            Fp32 m;
            m.fp = rand_sample(range);
            host_A[i*k + j] = m.fp;
            host_A_Fp64[i*k + j] = m.fp;
            Fp16 hi, lo, lofx;
            toFp32_F_FX(m.fp, &hi, &lo, &lofx);
            host_A_Hi[i*k + j] = hi;
            host_A_Lo[i*k + j] = lo;
            host_A_LoFx[i*k + j] = lofx;
            host_A_Hi32[i*k + j] = Fp16_To_Fp32(hi).fp;
            host_A_Lo32[i*k + j] = Fp16_To_Fp32(lo).fp;
            host_A_LoFx32[i*k + j] = Fp16_To_Fp32(lofx).fp;
        }
    }

    #ifdef debug
    fprintf(fp_debug, "A\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            fprintf(fp_debug, "%e ", host_A[i + j*m]);
        }
        fprintf(fp_debug, "\n");
    }
    #endif

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            Fp32 m;
            m.fp = rand_sample(range);
            host_B[i*n + j] = m.fp;
            host_B_Fp64[i*n + j] = m.fp;
            Fp16 hi, lo, lofx;
            toFp32_F_FX(m.fp, &hi, &lo, &lofx);
            host_B_Hi[i*n + j] = hi;
            host_B_Lo[i*n + j] = lo;
            host_B_LoFx[i*n + j] = lofx;
            host_B_Hi32[i*n + j] = Fp16_To_Fp32(hi).fp;
            host_B_Lo32[i*n + j] = Fp16_To_Fp32(lo).fp;
            host_B_LoFx32[i*n + j] = Fp16_To_Fp32(lofx).fp;
        }
    }

    #ifdef debug
    fprintf(fp_debug, "B\n");
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(fp_debug, "%e ", host_B[i + j*k]);
        }
        fprintf(fp_debug, "\n");
    }
    #endif

    const float alpha0 = 1.0f;
    const float alpha1 = pow(0.5f, 12);
    const float beta = 1.0f;
    const float zero = 0.0f;

    double* dev_A64;
    double* dev_B64;
    double* dev_C64;

    cudaMalloc((void**) &dev_A64, m * k * sizeof(double));
    cudaMalloc((void**) &dev_B64, n * k * sizeof(double));
    cudaMalloc((void**) &dev_C64, m * n * sizeof(double));

    cudaMemcpy(dev_A64, host_A_Fp64, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B64, host_B_Fp64, n * k * sizeof(double), cudaMemcpyHostToDevice);

    const double d_alpha = 1.0;
    const double d_zero = 0.0;

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &d_alpha, dev_A64, m, dev_B64, k, &d_zero, dev_C64, m);
    printf("Fp64\n");
    cudaMemcpy(host_C_Fp64, dev_C64, n * m * sizeof(double), cudaMemcpyDeviceToHost);
    
    #ifdef debug
    fprintf(fp_debug, "Fp64\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(fp_debug, "%e ", host_C_Fp64[i + j*m]);
        }
        fprintf(fp_debug, "\n");
    }
    #endif

    float* dev_A;
    float* dev_B;
    float* dev_C;

    cudaMalloc((void**) &dev_A, m * k * sizeof(float));
    cudaMalloc((void**) &dev_B, n * k * sizeof(float));
    cudaMalloc((void**) &dev_C, m * n * sizeof(float));

    cudaMemcpy(dev_A, host_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, host_B, n * k * sizeof(float), cudaMemcpyHostToDevice);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha0, dev_A, m, dev_B, k, &beta, dev_C, m);

    printf("Fp32\n");
    cudaMemcpy(host_C, dev_C, n * m * sizeof(float), cudaMemcpyDeviceToHost);
    memcpy(host_C_Fp32, host_C, sizeof(float) * m * n);
    
    #ifdef debug
    fprintf(fp_debug, "Fp32\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(fp_debug, "%e ", host_C_Fp32[i + j*m]);
        }
        fprintf(fp_debug, "\n");
    }
    #endif

    cudaMemset(dev_C, 0, sizeof(float) * m * n);

    float* dev_A_Hi32;
    float* dev_A_Lo32;
    float* dev_A_Lofx32;
    float* dev_B_Hi32;
    float* dev_B_Lo32;
    float* dev_B_Lofx32;

    cudaMalloc((void**)&dev_A_Hi32, sizeof(float) * m * k);
    cudaMalloc((void**)&dev_A_Lo32, sizeof(float) * m * k);
    //cudaMalloc((void**)&dev_A_Lofx32, sizeof(float) * m * k);
    cudaMalloc((void**)&dev_B_Hi32, sizeof(float) * n * k);
    cudaMalloc((void**)&dev_B_Lo32, sizeof(float) * n * k);
    //cudaMalloc((void**)&dev_B_Lofx32, sizeof(float) * n * k);
    

    cudaMemcpy(dev_A_Hi32, host_A_Hi32, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_A_Lo32, host_A_Lo32, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    //cudaMemcpy(dev_A_Lofx32, host_A_LoFx32, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B_Hi32, host_B_Hi32, sizeof(float) * n * k, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B_Lo32, host_B_Lo32, sizeof(float) * n * k, cudaMemcpyHostToDevice);
    //cudaMemcpy(dev_B_Lofx32, host_B_LoFx32, sizeof(float) * n * k, cudaMemcpyHostToDevice);
    
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha0, dev_A_Hi32, m, dev_B_Hi32, k, &beta, dev_C, m);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha1, dev_A_Hi32, m, dev_B_Lo32, k, &beta, dev_C, m);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha1, dev_A_Lo32, m, dev_B_Hi32, k, &beta, dev_C, m);
    //cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha1, dev_A_Hi32, m, dev_B_Lofx32, k, &beta, dev_C, m);
    //cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha1, dev_A_Lofx32, m, dev_B_Hi32, k, &beta, dev_C, m);

    printf("CUDA32\n");
    cudaMemcpy(host_C, dev_C, n * m * sizeof(float), cudaMemcpyDeviceToHost);
    memcpy(host_C_CUDA, host_C, sizeof(float) * m * n);
    
    #ifdef debug
    fprintf(fp_debug, "CUDA32\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(fp_debug, "%e ", host_C_CUDA[i + j*m]);
        }
        fprintf(fp_debug, "\n");
    }
    #endif
  
    memset(host_C, 0, sizeof(float) * m * n);
    stride_Gemm32_FX(m, n, k, 8, host_A, host_B, host_C);
    memcpy(host_C_Tensor_Stride, host_C, sizeof(float) * m * n);
    printf("TENSOR STRIDE\n");
    
    #ifdef debug
    fprintf(fp_debug, "TENSOR STRIDE\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(fp_debug, "%e ", host_C_Tensor_Stride[i + j*m]);
        }
        fprintf(fp_debug, "\n");
    }
    #endif

    cudaMemset(dev_C, 0, sizeof(float) * m * n);

    Fp16* dev_A_Hi;
    Fp16* dev_A_Lo;
    Fp16* dev_A_Lofx;
    Fp16* dev_B_Hi;
    Fp16* dev_B_Lo;
    Fp16* dev_B_Lofx;

    cudaMalloc((void**)&dev_A_Hi, sizeof(Fp16) * m * k);
    cudaMalloc((void**)&dev_A_Lo, sizeof(Fp16) * m * k);
    cudaMalloc((void**)&dev_A_Lofx, sizeof(Fp16) * m * k);
    cudaMalloc((void**)&dev_B_Hi, sizeof(Fp16) * n * k);
    cudaMalloc((void**)&dev_B_Lo, sizeof(Fp16) * n * k);
    cudaMalloc((void**)&dev_B_Lofx, sizeof(Fp16) * n * k);
    

    cudaMemcpy(dev_A_Hi, host_A_Hi, sizeof(Fp16) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_A_Lo, host_A_Lo, sizeof(Fp16) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_A_Lofx, host_A_LoFx, sizeof(Fp16) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B_Hi, host_B_Hi, sizeof(Fp16) * n * k, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B_Lo, host_B_Lo, sizeof(Fp16) * n * k, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B_Lofx, host_B_LoFx, sizeof(Fp16) * n * k, cudaMemcpyHostToDevice);

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha0, dev_A_Hi, CUDA_R_16F, m, dev_B_Hi, CUDA_R_16F, k, &beta, dev_C, CUDA_R_32F, m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha1, dev_A_Hi, CUDA_R_16F, m, dev_B_Lo, CUDA_R_16F, k, &beta, dev_C, CUDA_R_32F, m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha1, dev_A_Lo, CUDA_R_16F, m, dev_B_Hi, CUDA_R_16F, k, &beta, dev_C, CUDA_R_32F, m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    //cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha1, dev_A_Hi, CUDA_R_16F, m, dev_B_Lofx, CUDA_R_16F, k, &beta, dev_C, CUDA_R_32F, m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    //cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha1, dev_A_Lofx, CUDA_R_16F, m, dev_B_Hi, CUDA_R_16F, k, &beta, dev_C, CUDA_R_32F, m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    printf("TENSOR\n");
    cudaMemcpy(host_C, dev_C, n * m * sizeof(float), cudaMemcpyDeviceToHost);
    memcpy(host_C_Tensor, host_C, sizeof(float) * m * n);
    
    #ifdef debug
    fprintf(fp_debug, "TENSOR\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(fp_debug, "%e ", host_C_Tensor[i + j*m]);
        }
        fprintf(fp_debug, "\n");
    }
    #endif
    //print_matrix(m, n, host_C, n);
    //cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, (void*)dev_A_Hi, (void*)dev_A_Hi, CUDA_R_16F, 8, (void*)dev_B_Hi, CUDA_R_16F, 8, (void*)dev_B_Hi, (void*)dev_C, CUDA_R_16F, 8, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    double ratio_tensor = 0.0, origin, tensor_stride, tensor, cuda, ratio_stride = 0.0, ratio_fp32 = 0.0, ratio_cuda = 0.0;
    double mean_tensor_error = 0.0, mean_tensor_stride_error = 0.0, mean_ratio_tensor = 0.0, mean_ratio_tensor_stride = 0.0;
    int i_fp32, i_cuda, i_stride, i_tensor;
    for (int i = 0; i < m*n; i++) {
        
        // 误差比
        /*
        if (abs(host_C_Fp64[i] - (double)host_C_CUDA[i]) != 0.0 && abs(host_C_Fp64[i] - (double)host_C_Tensor[i]) / abs(host_C_Fp64[i] - (double)host_C_CUDA[i]) > ratio_tensor) {
            ratio_tensor = abs(host_C_Fp64[i] - (double)host_C_Tensor[i]) / abs(host_C_Fp64[i] - (double)host_C_CUDA[i]);
            origin = host_C_Fp64[i];
            tensor = host_C_Tensor[i];
            tensor_stride = host_C_Tensor_Stride[i];
            cuda = host_C_CUDA[i];
        }

         if (abs(host_C_Fp64[i] - (double)host_C_CUDA[i]) != 0.0 && abs(host_C_Fp64[i] - (double)host_C_Tensor_Stride[i]) / abs(host_C_Fp64[i] - (double)host_C_CUDA[i]) > ratio_stride) {
            ratio_stride = abs(host_C_Fp64[i] - (double)host_C_Tensor_Stride[i]) / abs(host_C_Fp64[i] - (double)host_C_CUDA[i]);
            origin = host_C_Fp64[i];
            tensor = host_C_Tensor[i];
            tensor_stride = host_C_Tensor_Stride[i];
            cuda = host_C_CUDA[i];
        }

        if (abs(host_C_Fp64[i] - (double)host_C_CUDA[i]) != 0.0) {
            mean_ratio_tensor += (abs(host_C_Fp64[i] - (double)host_C_Tensor[i]) / abs(host_C_Fp64[i] -(double) host_C_CUDA[i]));
            mean_ratio_tensor_stride += (abs(host_C_Fp64[i] - (double)host_C_Tensor_Stride[i]) / abs(host_C_Fp64[i] - (double)host_C_CUDA[i]));
        }*/

        //printf("%f %f %f %f %f\n", host_C_Fp64[i], host_C_Fp32[i], host_C_CUDA[i], host_C_Tensor[i], host_C_Tensor_Stride[i]);

        if (abs(host_C_Fp64[i] - (double)host_C_Fp32[i]) / abs(host_C_Fp64[i]) > ratio_fp32) {
            ratio_fp32 = abs(host_C_Fp64[i] - (double)host_C_Fp32[i]) / abs(host_C_Fp64[i]);
            i_fp32 = i;
        }
        if (abs(host_C_Fp64[i] - (double)host_C_CUDA[i]) / abs(host_C_Fp64[i]) > ratio_cuda) {
            ratio_cuda = abs(host_C_Fp64[i] - (double)host_C_CUDA[i]) / abs(host_C_Fp64[i]);
            i_cuda = i;
        }
        if (abs(host_C_Fp64[i] - (double)host_C_Tensor_Stride[i]) / abs(host_C_Fp64[i]) > ratio_stride) {
            ratio_stride = abs(host_C_Fp64[i] - (double)host_C_Tensor_Stride[i]) / abs(host_C_Fp64[i]);
            i_stride = i;
        }
        if (abs(host_C_Fp64[i] - (double)host_C_Tensor[i]) / abs(host_C_Fp64[i]) > ratio_tensor) {
            ratio_tensor = abs(host_C_Fp64[i] - (double)host_C_Tensor[i]) / abs(host_C_Fp64[i]);
            i_tensor = i;
        }
    }

    //printf("max_ratio: %e, origin: %e, cuda: %e, tensor: %e, tensor_stride: %e\n", ratio, origin, cuda, tensor, tensor_stride);
    printf("max_ratio_fp32: %e\n", ratio_fp32);
    printf("max_ratio_stride: %e\n", ratio_stride);
    printf("max_ratio_tensor: %e\n", ratio_tensor);
    printf("max_ratio_cuda: %e\n", ratio_cuda);
    printf("mean_tensor_ratio: %f, mean_tensor_stride_ratio: %f\n", mean_ratio_tensor / (m * n), mean_ratio_tensor_stride / (m * n));
    printf("mean_tensor_error: %f, mean_tensor_stride_error: %f\n", mean_tensor_error / (m * n), mean_tensor_stride_error / (m * n));

    

    FILE* fp_in = fopen("input", "a");

    find_input(fp_in, host_A, host_B, host_C_Fp32, host_C_Fp64, m, n, k, i_fp32, ratio_fp32);


    fprintf(fp, "%.2e/%.2e/%.2e/%.2e\n", ratio_fp32, ratio_cuda, ratio_tensor, ratio_stride);
    fclose(fp);
}

void GEMM16_TEST() {
    int m = 4, n = 1, k = 8;
    Fp16* host_A = (Fp16*)malloc(sizeof(Fp16) * m * k);
    Fp16* host_B = (Fp16*)malloc(sizeof(Fp16) * k * n);
    float* host_C = (float*)malloc(sizeof(float) * m * n);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            host_A[i*k + j] = Fp16{0x3C00};
        }
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            host_B[i*n + j] = Fp16{0x3C00};
        }
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            host_C[i*n + j] = 0.0f;
        }
    }

    Gemm16(m, n, k, host_A, host_B, host_C);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", host_C[i*n + j]);
        }
        printf("\n");
    }
    
    //printf("A*B + C: %e %x\n", Fp16_To_Fp32(host_C[i]).fp, host_C[i].us);
}

void CUDA_TENSOR_TEST() {
    cublasHandle_t handle = NULL;
    cublasCreate(&handle);

    int m = 4;
    int k = 8;
    int n = 8;

    Fp16* host_A = (Fp16*)malloc(sizeof(Fp16) * m * k);
    Fp16* host_B = (Fp16*)malloc(sizeof(Fp16) * n * k);
    float* host_C = (float*)malloc(sizeof(float) * m * n);

    memset(host_A, 0, sizeof(Fp16) * m * k);
    memset(host_B, 0, sizeof(Fp16) * n * k);
    memset(host_C, 0, sizeof(float) * m * n);

    Fp16 hi, lo, lofx;
    toFp32_F_FX(pow(0.5, 11) + pow(0.5, 12), &hi, &lo, &lofx);
    host_A[0] = hi;
    host_A[m] = Fp16{0x4000};
    toFp32_F_FX(pow(0.5, 12), &hi, &lo, &lofx);
    host_B[0] = hi;
    host_B[1] = Fp16{0x3C00};

    Fp16* dev_A;
    Fp16* dev_B;
    float* dev_C;

    cudaMalloc((void**) &dev_A, m * k * sizeof(Fp16));
    cudaMalloc((void**) &dev_B, n * k * sizeof(Fp16));
    cudaMalloc((void**) &dev_C, m * n * sizeof(float));

    cudaMemcpy(dev_A, host_A, m * k * sizeof(Fp16), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, host_B, n * k * sizeof(Fp16), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_C, host_C, m * n * sizeof(float), cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dev_A, CUDA_R_16F, m, dev_B, CUDA_R_16F, k, &beta, dev_C, CUDA_R_32F, m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    
    cudaMemcpy(host_C, dev_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 4; i++) printf("%e %x\n", host_C[i], Fp32{host_C[i]}.ui);

    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dev_A, CUDA_R_16F, m, dev_B, CUDA_R_16F, k, &beta, dev_C, CUDA_R_32F, m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaMemcpy(host_C, dev_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 4; i++) printf("%e %x\n", host_C[i], Fp32{host_C[i]}.ui);
}

int main(int argc, char* argv[]) {
    for (int i = 0; i < argc; i++) printf("%s ", argv[i]);
    printf("\n");
    TEST(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), (RANGE)atoi(argv[4]));
    return 0;
}