#include "roof.h"

#define BLOCKSIZE 1024

__global__ void saxpy(int n, double a, double *x, double *y)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}


extern "C"
void gpu_axpy(int n, double a, double b, double * x_in, double * y_in,
              struct roof_args *args)
{
    double *x, *y;
    size_t nbytes;

    int r, r_max;

    cudaEvent_t start, stop;
    double msec, sec;

    // Timer setup
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    nbytes = n * sizeof(double);
    cudaMalloc(&x, nbytes);
    cudaMalloc(&y, nbytes);

    cudaMemcpy(x, x_in, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(y, y_in, nbytes, cudaMemcpyHostToDevice);

    r_max = 1000;
    cudaEventRecord(start);
    for (r = 0; r < r_max; r++) {
        saxpy<<<1 + n / BLOCKSIZE, BLOCKSIZE>>>(n, a, x, y);
    }
    cudaEventRecord(stop);

    cudaMemcpy(y_in, y, nbytes, cudaMemcpyDeviceToHost);

    cudaFree(x);
    cudaFree(y);

    cudaEventElapsedTime(&msec, start, stop);
    sec = msec / 1000.f;

    args->runtime = sec;
    args->flops = 2. * r_max * n / sec;
    args->bw_load = 2. * r_max * nbytes / sec;
    args->bw_store = 1. * r_max * nbytes / sec;
}
