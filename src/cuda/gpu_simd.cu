#include "roof.h"

/* TODO: V100-specific numbers; how to generalize? */
#define NCORES 32
#define NBLOCKS 160
#define NTHREADS 128

__global__ void kadd(long r_max, float *sum)
{
    const float eps = 1e-6f;
    float reg[NCORES];
    long r;
    int i;

    for (i = 0; i < NCORES; i++)
        reg[i] = 1.f;

    for (r = 0; r < r_max; r++)
        for (i = 0; i < NCORES; i++)
            reg[i] = reg[i] + eps;

    *sum = 0.f;
    for (i = 0; i < NCORES; i++) *sum = *sum + reg[i];
}


__global__ void kmul(long r_max, float *sum)
{
    const float alpha = 1.f + 1e-6f;
    float reg[NCORES];
    long r;
    int i;

    for (i = 0; i < NCORES; i++)
        reg[i] = 1.f;

    for (r = 0; r < r_max; r++)
        for (i = 0; i < NCORES; i++)
            reg[i] = reg[i] * alpha;

    *sum = 0.f;
    for (i = 0; i < NCORES; i++) *sum = *sum + reg[i];
}


__global__ void kfma(long r_max, float *sum)
{
    const float eps = 1e-6f;
    const float alpha = 1.f + 1e-6f;
    float reg[NCORES];
    long r;
    int i;

    for (i = 0; i < NCORES; i++)
        reg[i] = 1.f;

    for (r = 0; r < r_max; r++)
        for (i = 0; i < NCORES; i++)
            reg[i] = alpha * reg[i] + eps;

    *sum = 0.f;
    for (i = 0; i < NCORES; i++) *sum = *sum + reg[i];
}


extern "C"
void gpu_add(void *args_in)
{
    struct roof_args *args;     // args
    cudaEvent_t start, stop;
    long r_max;
    float sum, *gpu_sum;
    float msec, runtime;

    args = (struct roof_args *) args_in;

    r_max = 1;
    cudaMalloc(&gpu_sum, sizeof(float));

    /* TODO: Move timer to kernel and use clock64() */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    *(args->runtime_flag) = 0;
    do {
        cudaEventRecord(start);
        kadd<<<NBLOCKS,NTHREADS>>>(r_max, gpu_sum);
        cudaEventRecord(stop);

        // Get results
        cudaMemcpy(&sum, gpu_sum, sizeof(float), cudaMemcpyDeviceToHost);

        cudaEventElapsedTime(&msec, start, stop);
        runtime = msec / 1000.f;

        if (runtime > args->min_runtime)
            *(args->runtime_flag) = 1;
        // TODO: Set mutex before write?

        // TODO: barrier?

        if (! *(args->runtime_flag)) r_max *= 2;
    } while (! *(args->runtime_flag));

    args->runtime = runtime;
    args->flops = (float) NBLOCKS * NTHREADS * NCORES * r_max / runtime;
    //args->flops = (float) 2 * NBLOCKS * NTHREADS * NCORES * r_max / runtime;
    args->bw_load = 0;
    args->bw_store = 0;

    cudaFree(gpu_sum);
}
