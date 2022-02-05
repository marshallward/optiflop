#include "roof.h"
#include <stdio.h>
#include "stopwatch.h"

#include <cublas.h>

#define MAXCORES 1
#define MAXTHREADS 64

extern "C"
void gpu_axpy_blas(int n, double a, double b, double * x_in, double * y_in,
                   struct roof_args *args)
{
    double *x, *y;
    size_t nbytes;

    long r_max;

    cudaEvent_t start, stop;
    cudaError_t error;
    float msec, sec;

    volatile double sum;

    // Timer setup
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    nbytes = n * sizeof(double);
    cudaMalloc(&x, nbytes);
    cudaMalloc(&y, nbytes);

    cudaMemcpy(x, x_in, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(y, y_in, nbytes, cudaMemcpyHostToDevice);

    /* Apparently cuBLAS setup time is horrendous, so we need to run it
       at least once before starting the time... */
    cublasDaxpy(n, a, x, 1, y, 1);

    r_max = 1;
    *(args->runtime_flag) = 0;
    do {
        cudaEventRecord(start);
        for (long r = 0; r < r_max; r++) {
            cublasDaxpy(n, a, x, 1, y, 1);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaMemcpy(y_in, y, nbytes, cudaMemcpyDeviceToHost);

        cudaEventElapsedTime(&msec, start, stop);
        sec = msec / 1000.f;

        if (sec > args->min_runtime)
            *(args->runtime_flag) = 1;
        else
            r_max *= 2;

    } while (!*(args->runtime_flag));

    //sum = 0.;
    //for (int i = 0; i < n; i++) sum += y_in[i];
    //if (sum != n) {
    //    printf("ERROR: Sum %f\n does not match!\n", sum);
    //    exit(1);
    //}

    cudaFree(x);
    cudaFree(y);

    args->runtime = sec;
    args->flops = 2. * r_max * n / sec;
    args->bw_load = 2. * r_max * nbytes / sec;
    args->bw_store = 1. * r_max * nbytes / sec;
}
