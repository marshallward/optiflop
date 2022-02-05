#include "bench.h"

void gpu_add(void *);
void gpu_fma(void *);
void gpu_axpy(int, double, double, double *, double *, struct roof_args *);
void gpu_axpy_blas(int, double, double, double *, double *, struct roof_args *);
void gpu_dgemm_blas(int, double, double, double *, double *, struct roof_args *);
