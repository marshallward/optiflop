#ifndef STOPWATCH_H_
#define STOPWATCH_H_

#include <time.h>   /* clockid_t, timespec */

/* If RAW clock is not present, replace with NTP-adjusted clock */
#ifndef CLOCK_MONOTONIC_RAW
#define CLOCK_MONOTONIC_RAW CLOCK_MONOTONIC
#endif


/* Stopwatch type */
enum stopwatch_backend {
    TIMER_UNDEF = -1,
    TIMER_STD = 0,
    TIMER_POSIX,
    TIMER_TSC,
    TIMER_MAX,
};


/* Stopwatch class */
typedef struct Stopwatch_struct {
    union stopwatch_context *context;

    void (*start)();
    void (*stop)();
    double (*runtime)();
    void (*destroy)();
} Stopwatch;


/* Context */
union stopwatch_context {
    struct stopwatch_context_std *tc_std;
    struct stopwatch_context_posix *tc_posix;
    struct stopwatch_context_tsc *tc_tsc;
};


/* Public Stopwatch methods */
Stopwatch * stopwatch_create(enum stopwatch_backend);


/* Public TSC-based methods */
double stopwatch_get_tsc_freq();
void stopwatch_set_tsc_freq();

#endif  // STOPWATCH_H_
