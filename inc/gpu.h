#include "bench.h"

void gpu_add(void *);
void gpu_axpy(int, SIMDTYPE, SIMDTYPE, SIMDTYPE *, SIMDTYPE *, struct roof_args *);
