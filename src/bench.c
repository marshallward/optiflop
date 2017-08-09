#include <pthread.h>    /* pthread_* */
#include <stdlib.h>     /* malloc, free */

#include "bench.h"

/* Thread control */
pthread_barrier_t timer_barrier;
pthread_mutex_t runtime_mutex;
volatile int runtime_flag;
