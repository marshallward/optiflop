#ifndef TIMER_H_
#define TIMER_H_

#include <pthread.h> /* pthread_* */
#include <stdint.h> /* uint64_t */
#include <time.h> 	/* clockid_t, timespec */

/* If RAW clock is not present, replace with NTP-adjusted clock */
/* TODO: Replace with some sort of kernel version check */
#ifndef CLOCK_MONOTONIC_RAW
#define CLOCK_MONOTONIC_RAW CLOCK_MONOTONIC
#endif

/* Timer definition */

enum stopwatch_type {
    TIMER_UNDEF = -1,
    TIMER_POSIX = 0,
    TIMER_TSC,
    TIMER_MAX,
};

union stopwatch_context_t {
    void *tc_untyped;
    struct stopwatch_context_posix_t *tc_posix;
    struct stopwatch_context_tsc_t *tc_tsc;
};

typedef struct Stopwatch_struct {
    union stopwatch_context_t context;

    void (*start)();
    void (*stop)();
    double (*runtime)();
    void (*destroy)();
} Stopwatch;

/* Generic Timer methods */

Stopwatch * stopwatch_create(enum stopwatch_type);

/* TSC Timer methods */

void stopwatch_init_tsc(Stopwatch *t);
void stopwatch_start_tsc(Stopwatch *t);
void stopwatch_stop_tsc(Stopwatch *t);
double stopwatch_runtime_tsc(Stopwatch *t);
void stopwatch_destroy_tsc(Stopwatch *t);

double stopwatch_get_tsc_freq();

/* POSIX Timer methods */

void stopwatch_init_posix(Stopwatch *t);
void stopwatch_start_posix(Stopwatch *t);
void stopwatch_stop_posix(Stopwatch *t);
double stopwatch_runtime_posix(Stopwatch *t);
void stopwatch_destroy_posix(Stopwatch *t);

#endif /* timer.h */
