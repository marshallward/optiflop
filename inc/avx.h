#ifndef OPTIFLOP_AVX_H_
#define OPTIFLOP_AVX_H_

#include "roof.h"

void avx_fma(void *);
void avx_fmac(void *);

/* TESTING??!? */
void simd_avx_add(struct roof_args *);
void simd_avx_mul(struct roof_args *);
void simd_avx_mac(struct roof_args *);


#endif  // OPTIFLOP_AVX_H_
