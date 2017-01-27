#ifndef _AVX_H_
#define _AVX_H_

/* Probably should be an input, not a global const... */
extern const uint64_t N;

double avx_add(void);
double avx_mac(void);

#endif
