#include "roof.h"
#include <stdio.h>

#define BLOCK_SIZE 16

//#define MAXCORES 8
//#define MAXTHREADS 1
#define MAXCORES 1
#define MAXTHREADS 64

__global__ void dgemm(int n, double *x, double *y)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0;
    if (col < n && row < n) {
        for (int i = 0; i < n; i++) {
            sum += x[row * n + i] * y[i * n + col];
        }
        y[row * n + col] = sum;
    }
}


extern "C"
void gpu_matmul(int n, double a, double b, double * x_in, double * y_in,
                struct roof_args *args)
{
    double *xm, *ym;
    double *x, *y;
    size_t nbytes;

    long r, r_max;
    //int nthreads, nblocks;

    cudaEvent_t start, stop;
    float msec, sec;

    volatile double sum;

    // Timer setup
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Use square matrices
    nbytes = (n * n) * sizeof(double);

    // We do not actually use x_in or y_in, but it seems not to matter much.
    // But it would not take much for a compiler to remove the calculation.
    xm = (double *) malloc(nbytes);
    ym = (double *) malloc(nbytes);
    for (int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            xm[i*n + j] = 0.;
            ym[i*n + j] = 0.;
        }
    }

    for (int i = 0; i < n; i++) {
        xm[i*n + i] = 1.;
        ym[i*n + i] = 1.;
    }

    //sum = 0.;
    //for (int i = 0; i < n; i++) {
    //    for (int j = 0; j < n; j++) {
    //        sum += xm[i*n +j];
    //    }
    //}
    //printf("xsum? %f\n", sum);

    //sum = 0.;
    //for (int i = 0; i < n; i++) {
    //    for (int j = 0; j < n; j++) {
    //        sum += ym[i*n +j];
    //    }
    //}
    //printf("ysum? %f\n", sum);

    cudaMalloc((void **) &x, nbytes);
    cudaMalloc((void **) &y, nbytes);

    cudaMemcpy(x, xm, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(y, ym, nbytes, cudaMemcpyHostToDevice);

    //nthreads = min(1 + (n - 1) / MAXCORES, MAXTHREADS);
    //nblocks = 1 + (n - 1) / (MAXTHREADS * MAXCORES);

    //printf("  ncores: %i\n", MAXCORES);
    //printf("nthreads: %i\n", nthreads);
    //printf(" nblocks: %i\n", nblocks);

    unsigned int grid_rows = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    r_max = 1;
    *(args->runtime_flag) = 0;
    do {
        cudaEventRecord(start);
        for (r = 0; r < r_max; r++) {
            dgemm<<<dimGrid,dimBlock>>>(n, x, y);
        }
        cudaEventRecord(stop);
        cudaMemcpy(ym, y, nbytes, cudaMemcpyDeviceToHost);

        cudaEventElapsedTime(&msec, start, stop);
        sec = msec / 1000.f;

        if (sec > args->min_runtime)
            *(args->runtime_flag) = 1;

        if (! *(args->runtime_flag)) r_max *= 2;
    } while (! *(args->runtime_flag));

    /* Do something with the sum */
    sum = 0.;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            sum += ym[i*n + j];
        }
    }
    //printf("sum? %f\n", sum);

    cudaEventElapsedTime(&msec, start, stop);
    sec = msec / 1000.f;

    //printf("r_max? %li\n", r_max);
    //printf("time? %f\n", sec);
    //printf("nbytes? %i\n", nbytes);

    cudaFree(x);
    cudaFree(y);

    free(xm);
    free(ym);

    double ops = 2. * n * n * n;
    //printf("ops? %f\n", ops);

    args->runtime = sec;
    args->flops = r_max * ops / sec;
    args->bw_load = 2. * r_max * nbytes / sec;
    args->bw_store = 1. * r_max * nbytes / sec;
}
