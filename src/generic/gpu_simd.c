#include "roof.h"

/* TODO: This is no longer the pthread, so we can pass roof_args directly. */
void gpu_add(void *args_in)
{
    struct roof_args *args;
    args = (struct roof_args *) args_in;

    args->runtime = 0.;
    args->flops = 0.;
    args->bw_load = 0.;
    args->bw_store = 0.;
}


void gpu_fma(void *args_in)
{
    struct roof_args *args;
    args = (struct roof_args *) args_in;

    args->runtime = 0.;
    args->flops = 0.;
    args->bw_load = 0.;
    args->bw_store = 0.;
}
