#ifndef _AVX_H_
#define _AVX_H_

/* Probably should be an input, not a global const... */
extern const uint64_t N;

double avx_add(void);
void * avx_add_thread(void *);

double avx_mac(void);
void * avx_mac_thread(void *);

#endif
