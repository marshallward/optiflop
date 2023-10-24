#include "roof.h"

void gpu_delsq(int n, double a, double b, double * x_in, double * y_in,
              struct roof_args *args)
{
    args->runtime = 0.;
    args->flops = 0.;
    args->bw_load = 0.;
    args->bw_store = 0.;
}
