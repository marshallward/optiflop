#include "gpu_roof.h"

__global__ void saxpy(int n, float a, float *x, float *y)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) y[i] = a * x[i] + y[i];
}

extern "C"
void ROOF_TEST(int n, float a, float b,
               float * restrict x_in, float * restrict y_in,
               struct roof_args *args)
{
    float *x, *y;

    Stopwatch *t;

    int r, r_max;
    int i;
    double runtime;

    cudaMalloc(&x, n);
    cudaMalloc(&y, n);

    cudaMemcpy(x, x_in, n, cudaMemcpyHostToDevice);
    cudaMemcpy(y, y_in, n, cudaMemcpyHostToDevice);

    saxpy<<(n + 255)/256, 256>>(n, a, x, y);

    cudaMemcpy(y, y_in, n, cudaMemcpyDeviceToHost);

    cudaFree(x);
    cudaFree(y);
}

#undef ROOF_TEST
#undef ROOF_KERNEL
#undef ROOF_FLOPS
#undef ROOF_BW_LOAD
#undef ROOF_BW_STORE
