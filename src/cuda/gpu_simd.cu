#include "roof.h"

/* TODO: V100-specific numbers; how to generalize? */
#define NCORES 32
#define NTHREADS 128
#define NBLOCKS 160


__global__ void kadd(long r_max, SIMDTYPE *sum, double *runtime)
{
    const SIMDTYPE eps = (SIMDTYPE) 1e-6;
    SIMDTYPE reg[NCORES];
    long r;
    int i;

    long long int start, end;

    for (i = 0; i < NCORES; i++)
        reg[i] = (SIMDTYPE) 1.;

    start = clock64();
    for (r = 0; r < r_max; r++)
        for (i = 0; i < NCORES; i++)
            reg[i] = reg[i] + eps;
    end = clock64();

    *runtime = (double) (end - start) / 1.230e9;

    *sum = (SIMDTYPE) 0.;
    for (i = 0; i < NCORES; i++) *sum = *sum + reg[i];
}


__global__ void kmul(long r_max, SIMDTYPE *sum)
{
    const SIMDTYPE alpha = (SIMDTYPE) (1. + 1e-6);
    SIMDTYPE reg[NCORES];
    long r;
    int i;

    for (i = 0; i < NCORES; i++)
        reg[i] = (SIMDTYPE) 1.;

    for (r = 0; r < r_max; r++)
        for (i = 0; i < NCORES; i++)
            reg[i] = reg[i] * alpha;

    *sum = (SIMDTYPE) 0.;
    for (i = 0; i < NCORES; i++) *sum = *sum + reg[i];
}


__global__ void kfma(long r_max, SIMDTYPE *sum)
{
    const SIMDTYPE eps = (SIMDTYPE) 1e-6;
    const SIMDTYPE alpha = (SIMDTYPE) (1. + 1e-6);
    SIMDTYPE reg[NCORES];
    long r;
    int i;

    for (i = 0; i < NCORES; i++)
        reg[i] = 1.f;

    for (r = 0; r < r_max; r++)
        for (i = 0; i < NCORES; i++)
            reg[i] = alpha * reg[i] + eps;

    *sum = (SIMDTYPE) 0.;
    for (i = 0; i < NCORES; i++) *sum = *sum + reg[i];
}


extern "C"
void gpu_add(void *args_in)
{
    struct roof_args *args;     // args
    cudaEvent_t start, stop;
    long r_max;
    SIMDTYPE sum, *gpu_sum;
    float msec, runtime;

    // testing
    double *gpu_runtime;
    double new_runtime;

    args = (struct roof_args *) args_in;

    // TODO: useful?
    //cudaDeviceReset();

    r_max = 1;
    cudaMalloc(&gpu_sum, sizeof(SIMDTYPE));
    cudaMalloc(&gpu_runtime, sizeof(double));

    /* TODO: Move timer to kernel and use clock64() */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    *(args->runtime_flag) = 0;
    do {
        cudaEventRecord(start);
        kadd<<<NBLOCKS,NTHREADS>>>(r_max, gpu_sum, gpu_runtime);
        cudaEventRecord(stop);

        // Get results
        cudaMemcpy(&sum, gpu_sum, sizeof(SIMDTYPE), cudaMemcpyDeviceToHost);
        cudaMemcpy(&new_runtime, gpu_runtime, sizeof(double), cudaMemcpyDeviceToHost);

        cudaEventElapsedTime(&msec, start, stop);
        runtime = msec / 1000.f;

        if (runtime > args->min_runtime)
            *(args->runtime_flag) = 1;

        // TODO: Set mutex before write?

        // TODO: barrier?

        if (! *(args->runtime_flag)) r_max *= 2;
    } while (! *(args->runtime_flag));

    args->runtime = runtime;
    args->flops = (double) NBLOCKS * NTHREADS * NCORES * r_max / runtime;
    args->bw_load = 0;
    args->bw_store = 0;

    cudaFree(gpu_sum);
    cudaFree(gpu_runtime);
}
