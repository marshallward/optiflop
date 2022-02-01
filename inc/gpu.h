#include "bench.h"

void gpu_add(void *);
void gpu_fma(void *);
void gpu_axpy(int, double, double, double *, double *, struct roof_args *);
void gpu_matmul(int, double, double, double *, double *, struct roof_args *);
