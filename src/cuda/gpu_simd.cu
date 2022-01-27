#include "roof.h"

/* TODO: V100-specific numbers; how to generalize? */
#define NCORES 32
#define NTHREADS 128
#define NBLOCKS 160


__global__ void kadd(long r_max, double *sum, double *runtime)
{
    const double eps = (double) 1e-6;
    double reg[NCORES];
    long long int start, end;

    long r;
    int i;

    for (i = 0; i < NCORES; i++)
        reg[i] = (double) 1.;

    start = clock64();
    for (r = 0; r < r_max; r++)
        for (i = 0; i < NCORES; i++)
            reg[i] = reg[i] + eps;
    end = clock64();

    *runtime = (double) (end - start) / 1.230e9;

    *sum = (double) 0.;
    for (i = 0; i < NCORES; i++) *sum = *sum + reg[i];
}


__global__ void kmul(long r_max, double *sum, double *runtime)
{
    const double alpha = (double) (1. + 1e-6);
    double reg[NCORES];
    long long int start, end;

    long r;
    int i;

    for (i = 0; i < NCORES; i++)
        reg[i] = (double) 1.;

    start = clock64();
    for (r = 0; r < r_max; r++)
        for (i = 0; i < NCORES; i++)
            reg[i] = reg[i] * alpha;
    end = clock64();

    *runtime = (double) (end - start) / 1.230e9;

    *sum = (double) 0.;
    for (i = 0; i < NCORES; i++) *sum = *sum + reg[i];
}


__global__ void kfma(long r_max, double *sum, double *runtime)
{
    const double eps = (double) 1e-6;
    const double alpha = (double) (1. + 1e-6);
    double reg[NCORES];
    long long int start, end;

    long r;
    int i;

    for (i = 0; i < NCORES; i++)
        reg[i] = 1.f;

    start = clock64();
    for (r = 0; r < r_max; r++)
        for (i = 0; i < NCORES; i++)
            reg[i] = alpha * reg[i] + eps;
    end = clock64();

    *runtime = (double) (end - start) / 1.230e9;

    *sum = (double) 0.;
    for (i = 0; i < NCORES; i++) *sum = *sum + reg[i];
}


extern "C"
void gpu_add(void *args_in)
{
    struct roof_args *args;     // args
    cudaEvent_t start, stop;
    long r_max;
    double sum, *gpu_sum;
    float msec, runtime;

    // testing
    double *gpu_runtime;
    double new_runtime;

    args = (struct roof_args *) args_in;

    // TODO: useful?
    //cudaDeviceReset();

    r_max = 1;
    cudaMalloc(&gpu_sum, sizeof(double));
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
        cudaMemcpy(&sum, gpu_sum, sizeof(double), cudaMemcpyDeviceToHost);
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


/* TODO: Merge with gpu_add via kernel pointer? */
extern "C"
void gpu_fma(void *args_in)
{
    struct roof_args *args;     // args
    cudaEvent_t start, stop;
    long r_max;
    double sum, *gpu_sum;
    float msec, runtime;

    // testing
    double *gpu_runtime;
    double new_runtime;

    args = (struct roof_args *) args_in;

    // TODO: useful?
    //cudaDeviceReset();

    r_max = 1;
    cudaMalloc(&gpu_sum, sizeof(double));
    cudaMalloc(&gpu_runtime, sizeof(double));

    /* TODO: Move timer to kernel and use clock64() */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    *(args->runtime_flag) = 0;
    do {
        cudaEventRecord(start);
        kfma<<<NBLOCKS,NTHREADS>>>(r_max, gpu_sum, gpu_runtime);
        cudaEventRecord(stop);

        // Get results
        cudaMemcpy(&sum, gpu_sum, sizeof(double), cudaMemcpyDeviceToHost);
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
    args->flops = (double) 2 * NBLOCKS * NTHREADS * NCORES * r_max / runtime;
    args->bw_load = 0;
    args->bw_store = 0;

    cudaFree(gpu_sum);
    cudaFree(gpu_runtime);
}
