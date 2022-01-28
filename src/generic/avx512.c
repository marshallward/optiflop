#include <pthread.h>    /* pthread_* */

#include "roof.h"


void avx512_add(void *args_in)
{
    struct roof_args *args;

    args = (struct roof_args *) args_in;

    args->runtime = 0.;
    args->flops = 0.;
    args->bw_load = 0.;
    args->bw_store = 0.;
}


void avx512_fma(void *args_in)
{
    struct roof_args *args;

    args = (struct roof_args *) args_in;

    args->runtime = 0.;
    args->flops = 0.;
    args->bw_load = 0.;
    args->bw_store = 0.;
}


void avx512_fmac(void *args_in)
{
    struct roof_args *args;

    args = (struct roof_args *) args_in;

    args->runtime = 0.;
    args->flops = 0.;
    args->bw_load = 0.;
    args->bw_store = 0.;
}
