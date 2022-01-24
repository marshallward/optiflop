#include <pthread.h>    /* pthread_* */

#include "roof.h"

void avx_add(void *args_in)
{
    struct roof_args *args;

    args = (struct roof_args *) args_in;

    args->runtime = 0.;
    args->flops = 0.;
    args->bw_load = 0.;
    args->bw_store = 0.;
}


void avx_mac(void *args_in)
{
    struct roof_args *args;

    args = (struct roof_args *) args_in;

    args->runtime = 0.;
    args->flops = 0.;
    args->bw_load = 0.;
    args->bw_store = 0.;
}
