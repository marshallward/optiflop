#ifndef AVX_H_
#define AVX_H_

#include <pthread.h>    /* pthread_* */

#include "flop.h"   /* bench_arg_t */

void avx_add(bench_arg_t *);
void avx_mac(bench_arg_t *);

extern pthread_barrier_t timer_barrier;
extern pthread_mutex_t runtime_mutex;
#endif
