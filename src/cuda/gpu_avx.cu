#include "roof.h"

// testing...
#include <unistd.h> // sleep

extern "C"
void gpu_avx(void *args_in)
{
    /* Thread input */
    struct roof_args *args;

    cudaEvent_t start, stop;
    float msec, sec;

    args = (struct roof_args *) args_in;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    /* TODO: Actual work! */
    sleep(1);
    cudaEventRecord(stop);

    cudaEventElapsedTime(&msec, start, stop);
    sec = msec / 1000.f;

    /* TODO: Actual work! */
    args->runtime = sec;
    args->flops = 0.;
    args->bw_load = 0;
    args->bw_store =0;
}
