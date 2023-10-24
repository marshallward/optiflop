#include "roof.h"
#include <stdio.h>

#define MAXCORES 1
#define MAXTHREADS 64

__global__ void delsq(int n, double a, double b, double *x, double *y)
{
    int i0 = MAXCORES * (blockDim.x * blockIdx.x + threadIdx.x);
    for (int i = i0; i < min(i0 + MAXCORES, n); i++)
        //if (i > 0 && i < n-1)
        //    y[i] = a * x[i] + b * (x[i-1] + x[i+1]);
        //if (i > 8 && i < n-8)
        //    y[i] = a * x[i] + b * (x[i-8] + x[i+8]);
        if (i > 0 && i < n-1)
            y[i] = a * x[i] + a * x[i+1];
}


__global__ void copy(int n, double a, double b, double *x, double *y)
{
    int i0 = MAXCORES * (blockDim.x * blockIdx.x + threadIdx.x);
    for (int i = i0; i < min(i0 + MAXCORES, n); i++)
        //x[i] = y[i];
        //y[i] = x[i] + y[i];
        y[i] = a * x[i];
}


extern "C"
void gpu_delsq(int n, double a, double b, double * x_in, double * y_in,
              struct roof_args *args)
{
    double *x, *y;
    size_t nbytes;

    long r_max;
    int nthreads, nblocks;

    cudaEvent_t start, stop;
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

    nthreads = min(1 + (n - 1) / MAXCORES, MAXTHREADS);
    nblocks = 1 + (n - 1) / (MAXTHREADS * MAXCORES);

    //printf("  ncores: %i\n", MAXCORES);
    //printf("nthreads: %i\n", nthreads);
    //printf(" nblocks: %i\n", nblocks);

    r_max = 1;
    *(args->runtime_flag) = 0;
    do {
        cudaEventRecord(start);
        for (long r = 0; r < r_max; r++) {
            delsq<<<nblocks,nthreads>>>(n, a, b, x, y);
            //copy<<<nblocks,nthreads>>>(n, a, b, x, y);
        }
        cudaEventRecord(stop);
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

    cudaEventElapsedTime(&msec, start, stop);
    sec = msec / 1000.f;

    args->runtime = sec;
    //args->flops = 4. * r_max * n / sec;
    //args->bw_load = 1. * r_max * nbytes / sec;
    //args->bw_store = 1. * r_max * nbytes / sec;
    args->flops = 2. * r_max * n / sec;
    args->bw_load = 1. * r_max * nbytes / sec;
    args->bw_store = 1. * r_max * nbytes / sec;
}
