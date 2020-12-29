#include <stdlib.h>     /* malloc, free */
#include <time.h>       /* clock[id]_* */

#include "stopwatch.h"
#include "stopwatch_posix.h"


/* POSIX Stopwatch methods */

struct stopwatch_context_posix {
    clockid_t clock;
    struct timespec ts_start, ts_end;
};


void stopwatch_init_posix(Stopwatch *t)
{
    t->context = malloc(sizeof(union stopwatch_context));
    t->context->tc_posix = malloc(sizeof(struct stopwatch_context_posix));
    t->context->tc_posix->clock = CLOCK_MONOTONIC_RAW;
}


void stopwatch_start_posix(Stopwatch *t)
{
    clock_gettime(t->context->tc_posix->clock,
                  &(t->context->tc_posix->ts_start));
}


void stopwatch_stop_posix(Stopwatch *t)
{
    clock_gettime(t->context->tc_posix->clock,
                  &(t->context->tc_posix->ts_end));
}


double stopwatch_runtime_posix(Stopwatch *t)
{
    return (double) (t->context->tc_posix->ts_end.tv_sec
                            - t->context->tc_posix->ts_start.tv_sec)
            + (double) (t->context->tc_posix->ts_end.tv_nsec
                            - t->context->tc_posix->ts_start.tv_nsec) / 1e9;
}


void stopwatch_destroy_posix(Stopwatch *t)
{
    free(t->context->tc_posix);
    free(t->context);
    free(t);
}
