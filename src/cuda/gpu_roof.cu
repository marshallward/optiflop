#include "roof.h"
#include <stdio.h>

#define MAXCORES 2
#define MAXTHREADS 800


__global__ void saxpy(int n, double a, double *x, double *y)
{
    int i0 = MAXCORES * (blockDim.x * blockIdx.x + threadIdx.x);
    for (int i = i0; i < min(i0 + MAXCORES, n); i++)
        y[i] = a * x[i] + y[i];
}


extern "C"
void gpu_axpy(int n, double a, double b, double * x_in, double * y_in,
              struct roof_args *args)
{
    double *x, *y;
    size_t nbytes;

    int r, r_max;
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
    cudaEventRecord(start);
    for (r = 0; r < r_max; r++) {
        saxpy<<<nblocks,nthreads>>>(n, a, x, y);
    }
    cudaEventRecord(stop);

    cudaMemcpy(y_in, y, nbytes, cudaMemcpyDeviceToHost);

    /* Not yet confident this is working, so check the sum. */
    /* Also ensures that the value is touched and won't be optimized out. */
    /* TODO: Later, we can rely on `volatile` and drop this sum. */
    sum = 0.;
    for (int i = 0; i < n; i++) sum += y_in[i];
    if (sum != 4. * n) {
        printf("ERROR: Sum (%f\n does not match!\n", sum);
        exit(1);
    }

    cudaFree(x);
    cudaFree(y);

    cudaEventElapsedTime(&msec, start, stop);
    sec = msec / 1000.f;

    args->runtime = sec;
    args->flops = 2. * r_max * n / sec;
    args->bw_load = 2. * r_max * nbytes / sec;
    args->bw_store = 1. * r_max * nbytes / sec;
}
