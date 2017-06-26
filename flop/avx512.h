#ifndef FLOP_AVX512_H_
#define FLOP_AVX512_H_

#include <pthread.h>    /* pthread_* */

void * avx512_add(void *);
void * avx512_fma(void *);

extern pthread_barrier_t timer_barrier;
extern pthread_mutex_t runtime_mutex;
#endif  // FLOP_AVX512_H_
