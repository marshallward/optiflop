#ifndef TIMER_H_
#define TIMER_H_

#include <pthread.h> /* pthread_* */
#include <stdint.h> /* uint64_t */
#include <time.h> 	/* clockid_t, timespec */

/* Timer definition */

typedef enum _stopwatch_t {
    TIMER_UNDEF = -1,
    TIMER_POSIX = 0,
    TIMER_TSC,
    TIMER_MAX,
} TimerType;


typedef struct _stopwatch_context_tsc_t {
    double cpufreq;
    uint64_t rax0, rdx0, rax1, rdx1;
} stopwatch_context_tsc_t;


typedef struct _stopwatch_context_posix_t {
    clockid_t clock;
    struct timespec ts_start, ts_end;
} stopwatch_context_posix_t;


typedef union _stopwatch_context_t {
    void *tc_untyped;
    stopwatch_context_posix_t *tc_posix;
    stopwatch_context_tsc_t *tc_tsc;
} stopwatch_context_t;


typedef struct _Stopwatch {
    stopwatch_context_t context;

    void (*start)();
    void (*stop)();
    double (*runtime)();
    void (*destroy)();
} Stopwatch;


/* Generic Timer methods */

Stopwatch * stopwatch_create(TimerType);

/* TSC Timer methods */

void stopwatch_init_tsc(Stopwatch *t);
void stopwatch_start_tsc(Stopwatch *t);
void stopwatch_stop_tsc(Stopwatch *t);
double stopwatch_runtime_tsc(Stopwatch *t);
void stopwatch_destroy_tsc(Stopwatch *t);

/* TSC support functions */

double stopwatch_get_tsc_freq(Stopwatch *t);

/* POSIX Timer methods */

void stopwatch_init_posix(Stopwatch *t);
void stopwatch_start_posix(Stopwatch *t);
void stopwatch_stop_posix(Stopwatch *t);
double stopwatch_runtime_posix(Stopwatch *t);
void stopwatch_destroy_posix(Stopwatch *t);

#endif /* timer.h */
