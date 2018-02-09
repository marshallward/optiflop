#ifndef STOPWATCH_H_
#define STOPWATCH_H_

#include <stdint.h> /* uint64_t */
#include <time.h>   /* clockid_t, timespec */

/* If RAW clock is not present, replace with NTP-adjusted clock */
/* TODO: Replace with some sort of kernel version check */
#ifndef CLOCK_MONOTONIC_RAW
#define CLOCK_MONOTONIC_RAW CLOCK_MONOTONIC
#endif

/* Stopwatch type */
enum stopwatch_backend {
    TIMER_UNDEF = -1,
    TIMER_POSIX = 0,
    TIMER_TSC,
    TIMER_MAX,
};

/* Stopwatch class */
typedef struct Stopwatch_struct {
    union stopwatch_context_t *context;

    void (*start)();
    void (*stop)();
    double (*runtime)();
    void (*destroy)();
} Stopwatch;

/* Context */
union stopwatch_context_t {
    struct stopwatch_context_posix_t *tc_posix;
    struct stopwatch_context_tsc_t *tc_tsc;
};

/* TSC frequency (if available) */
extern double tsc_freq;

/* Generic Timer methods */
Stopwatch * stopwatch_create(enum stopwatch_backend);

/* POSIX Timer methods */
void stopwatch_init_posix(Stopwatch *t);
void stopwatch_start_posix(Stopwatch *t);
void stopwatch_stop_posix(Stopwatch *t);
double stopwatch_runtime_posix(Stopwatch *t);
void stopwatch_destroy_posix(Stopwatch *t);

/* TSC Timer methods */
void stopwatch_init_tsc(Stopwatch *t);
void stopwatch_start_tsc(Stopwatch *t);
void stopwatch_stop_tsc(Stopwatch *t);
double stopwatch_runtime_tsc(Stopwatch *t);
void stopwatch_destroy_tsc(Stopwatch *t);
double stopwatch_get_tsc_freq();
void stopwatch_set_tsc_freq();

#endif  // STOPWATCH_H_
