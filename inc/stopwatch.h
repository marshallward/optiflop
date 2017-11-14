#ifndef FLOP_STOPWATCH_H_
#define FLOP_STOPWATCH_H_

#include <stdint.h> /* uint64_t */
#include <time.h>   /* clockid_t, timespec */

/* If RAW clock is not present, replace with NTP-adjusted clock */
/* TODO: Replace with some sort of kernel version check */
#ifndef CLOCK_MONOTONIC_RAW
#define CLOCK_MONOTONIC_RAW CLOCK_MONOTONIC
#endif

/* Timer definition */

enum stopwatch_backend {
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

/* Context definitions */
/* TODO: Can I move these out of the header? */

struct stopwatch_context_tsc_t {
    double cpufreq;
    uint64_t rax0, rdx0, rax1, rdx1;
};

struct stopwatch_context_posix_t {
    clockid_t clock;
    struct timespec ts_start, ts_end;
};

/* Generic Timer methods */

Stopwatch * stopwatch_create(enum stopwatch_backend);

/* TSC Timer methods */

void stopwatch_init_tsc(Stopwatch *t);
void stopwatch_start_tsc(Stopwatch *t);
void stopwatch_stop_tsc(Stopwatch *t);
double stopwatch_runtime_tsc(Stopwatch *t);
void stopwatch_destroy_tsc(Stopwatch *t);

/* POSIX Timer methods */

void stopwatch_init_posix(Stopwatch *t);
void stopwatch_start_posix(Stopwatch *t);
void stopwatch_stop_posix(Stopwatch *t);
double stopwatch_runtime_posix(Stopwatch *t);
void stopwatch_destroy_posix(Stopwatch *t);

#endif  // FLOP_STOPWATCH_H_
