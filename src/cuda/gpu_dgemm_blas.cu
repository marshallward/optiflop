#include <cublas_v2.h>
#include <time.h>
#include "roof.h"

#include <stdio.h>


extern "C"
void gpu_dgemm_blas(int n, double a, double b, double * x_in, double * y_in,
                    struct roof_args *args)
{
    cublasHandle_t handle;
    cudaStream_t stream;
    cublasOperation_t op = CUBLAS_OP_N;
    size_t nbytes;

    double *A, *B, *C;
    double *dA, *dB, *dC;
    double alpha = 1.;
    double beta = 0.;

    long r_max;

    clockid_t clock = CLOCK_MONOTONIC_RAW;
    struct timespec start, stop;
    double sec;

    volatile double sum;

    // Use square matrices
    nbytes = (n * n) * sizeof(double);
    A = (double *) malloc(nbytes);
    B = (double *) malloc(nbytes);
    C = (double *) malloc(nbytes);

    // We do not actually use x_in or y_in, but it seems not to matter much.
    // But it would not take much for a compiler to remove the calculation.
    for (int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            //A[i*n+j] = (i == j) ? 1. : 0.;
            //B[i*n+j] = (i == j) ? 1. : 0.;
            A[i*n+j] = (double) (i*n+j) * 1e-3;
            B[i*n+j] = (double) (j*n+i) * 1e-3;
        }
    }
    cudaMalloc((void **) &dA, nbytes);
    cudaMalloc((void **) &dB, nbytes);
    cudaMalloc((void **) &dC, nbytes);

    cublasSetMatrix(n, n, sizeof(double), A, n, dA, n);
    cublasSetMatrix(n, n, sizeof(double), B, n, dB, n);

    cublasCreate(&handle);
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);

    /* Rev up the core */
    cublasDgemm(handle, op, op, n, n, n, &alpha, dA, n, dB, n, &beta, dC, n);

    r_max = 1;
    *(args->runtime_flag) = 0;
    do {
        cudaStreamSynchronize(stream);
        clock_gettime(clock, &start);
        for (long r = 0; r < r_max; r++) {
            cublasDgemm(handle, op, op, n, n, n, &alpha, dA, n, dB, n, &beta, dC, n);
        }
        cudaStreamSynchronize(stream);
        clock_gettime(clock, &stop);

        sec = (double) (stop.tv_sec - start.tv_sec)
            + (double) (stop.tv_nsec - start.tv_nsec) * 1e-9;

        if (sec > args->min_runtime)
            *(args->runtime_flag) = 1;
        else
            r_max *= 2;

    } while (!*(args->runtime_flag));

    /* Do something with the sum */
    cublasGetMatrix(n, n, sizeof(double), dC, n, C, n);

    sum = 0.;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            sum += C[i*n + j];
        }
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    free(A);
    free(B);
    free(C);

    double ops = 2. * n * n * n;

    args->runtime = sec;
    args->flops = r_max * ops / sec;
    args->bw_load = 2. * r_max * nbytes / sec;
    args->bw_store = 1. * r_max * nbytes / sec;
}
