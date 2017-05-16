#ifndef AVX_H_
#define AVX_H_

#include <pthread.h>    /* pthread_* */

void avx_add(double *, double *);
void avx_mac(double *, double *);

extern pthread_barrier_t timer_barrier;
extern pthread_mutex_t runtime_mutex;
#endif
