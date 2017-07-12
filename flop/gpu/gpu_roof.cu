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
    dim3 threadBlockRows(256, 1);

    size_t nbytes;

    nbytes = n * sizeof(float);

    cudaMalloc(&x, nbytes);
    cudaMalloc(&y, nbytes);

    cudaMemcpy(x, x_in, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(y, y_in, nbytes, cudaMemcpyHostToDevice);

    saxpy<<<(n + 255)/256, 256>>>(n, a, x, y);

    cudaMemcpy(y_in, y, nbytes, cudaMemcpyDeviceToHost);

    cudaFree(x);
    cudaFree(y);
}
