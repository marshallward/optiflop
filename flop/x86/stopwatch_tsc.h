#ifndef STOPWATCH_TSC_H_
#define STOPWATCH_TSC_H_

#include <pthread.h> /* pthread_* */
#include <stdint.h> /* uint64_t */
#include <time.h> 	/* clockid_t, timespec */

/* Context */

struct stopwatch_context_tsc_t {
    double cpufreq;
    uint64_t rax0, rdx0, rax1, rdx1;
};

/* TSC Timer methods */

void stopwatch_init_tsc(Stopwatch *t);
void stopwatch_start_tsc(Stopwatch *t);
void stopwatch_stop_tsc(Stopwatch *t);
double stopwatch_runtime_tsc(Stopwatch *t);
void stopwatch_destroy_tsc(Stopwatch *t);

double stopwatch_get_tsc_freq();

#endif /* STOPWATCH_TSC_H_ */
