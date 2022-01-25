#include "roof.h"

void gpu_axpy(int n, SIMDTYPE a, SIMDTYPE b, SIMDTYPE * x_in, SIMDTYPE * y_in,
              struct roof_args *args)
{
    args->runtime = 0.;
    args->flops = 0.;
    args->bw_load = 0.;
    args->bw_store = 0.;
}
