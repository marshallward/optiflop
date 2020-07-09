#ifndef FLOP_AVX_FMA_H_
#define FLOP_AVX_FMA_H_

#include <pthread.h>    /* pthread_* */

void * avx_fma(void *);
void * avx_fmac(void *);

extern pthread_barrier_t timer_barrier;
extern pthread_mutex_t runtime_mutex;
#endif  // FLOP_AVX_FMA_H_
