#ifndef FLOP_AVX_H_
#define FLOP_AVX_H_

#include <pthread.h>    /* pthread_* */

void * avx_add(void *);
void * avx_mac(void *);
void * avx_fma(void *);

extern pthread_barrier_t timer_barrier;
extern pthread_mutex_t runtime_mutex;
#endif  // FLOP_AVX_H_
