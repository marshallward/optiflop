#ifndef AVX512_H_
#define AVX512_H_

#include <pthread.h>    /* pthread_* */

void * avx_fma(void *);

extern pthread_barrier_t timer_barrier;
extern pthread_mutex_t runtime_mutex;
#endif
