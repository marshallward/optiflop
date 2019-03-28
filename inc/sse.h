#ifndef FLOP_SSE_H_
#define FLOP_SSE_H_

#include <pthread.h>    /* pthread_* */

void * sse_add(void *);

extern pthread_barrier_t timer_barrier;
extern pthread_mutex_t runtime_mutex;
#endif  // FLOP_SSE_H_
