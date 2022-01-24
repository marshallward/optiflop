#ifndef OPTIFLOP_AVX512_H_
#define OPTIFLOP_AVX512_H_

#ifndef SIMDTYPE
#define SIMDTYPE float
#endif

void avx512_add(void *);
void avx512_fma(void *);

#endif  // OPTIFLOP_AVX512_H_
