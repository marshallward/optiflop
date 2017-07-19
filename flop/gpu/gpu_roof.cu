#include "roof.h"
#include "gpu_roof.h"


__global__ void saxpy(int n, float a, float *x, float *y)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) y[i] = a * x[i] + y[i];
}


extern "C"
void gpu_axpy(int n, float a, float b, float * x_in, float * y_in,
              struct roof_args *args)
{
    float *x, *y;
    size_t nbytes;

    int r, r_max;

    cudaEvent_t start, stop;
    float msec, sec;

    // Timer setup
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    nbytes = n * sizeof(float);
    cudaMalloc(&x, nbytes);
    cudaMalloc(&y, nbytes);

    cudaMemcpy(x, x_in, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(y, y_in, nbytes, cudaMemcpyHostToDevice);

    r_max = 1000;
    cudaEventRecord(start);
    for (r = 0; r < r_max; r++) {
        saxpy<<<1, 1024>>>(n, a, x, y);
    }
    cudaEventRecord(stop);

    cudaMemcpy(y_in, y, nbytes, cudaMemcpyDeviceToHost);

    cudaFree(x);
    cudaFree(y);

    cudaEventElapsedTime(&msec, start, stop);
    sec = msec / 1000.f;

    args->runtime = sec;
    args->flops = 2 * r_max * n / sec;
    args->bw_load = 2 * r_max * nbytes / sec;
    args->bw_store = nbytes / sec;
}
