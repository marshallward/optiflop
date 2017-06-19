#include <stdint.h>     /* uint64_t */
#include <stdlib.h>     /* malloc, free */
#include <time.h>       /* clock[id]_* */
#include <stdio.h>

#include "stopwatch.h"
#include "x86/stopwatch_tsc.h"

/* Method lookup tables */

void (*stopwatch_init_funcs[TIMER_MAX])(Stopwatch *t) = {
    stopwatch_init_posix,
    stopwatch_init_tsc,
};

void (*stopwatch_start_funcs[TIMER_MAX])(Stopwatch *t) = {
    stopwatch_start_posix,
    stopwatch_start_tsc,
};

void (*stopwatch_stop_funcs[TIMER_MAX])(Stopwatch *t) = {
    stopwatch_stop_posix,
    stopwatch_stop_tsc,
};

double (*stopwatch_runtime_funcs[TIMER_MAX])(Stopwatch *t) = {
    stopwatch_runtime_posix,
    stopwatch_runtime_tsc,
};

void (*stopwatch_destroy_funcs[TIMER_MAX])(Stopwatch *t) = {
    stopwatch_destroy_posix,
    stopwatch_destroy_tsc,
};

/* Context definitions */

struct stopwatch_context_posix_t {
    clockid_t clock;
    struct timespec ts_start, ts_end;
};

const size_t stopwatch_context_size[TIMER_MAX] = {
    sizeof(struct stopwatch_context_posix_t),
    sizeof(struct stopwatch_context_tsc_t),
};

/* Generic Stopwatch methods */

Stopwatch * stopwatch_create(enum stopwatch_type type)
{
    Stopwatch *t;

    t = malloc(sizeof(Stopwatch));

    t->start = stopwatch_start_funcs[type];
    t->stop = stopwatch_stop_funcs[type];
    t->runtime = stopwatch_runtime_funcs[type];
    t->destroy = stopwatch_destroy_funcs[type];

    stopwatch_init_funcs[type](t);

    return t;
}

/* POSIX Stopwatch methods */

void stopwatch_init_posix(Stopwatch *t)
{
    t->context.tc_posix = malloc(sizeof(struct stopwatch_context_posix_t));
    t->context.tc_posix->clock = CLOCK_MONOTONIC_RAW;
}

void stopwatch_start_posix(Stopwatch *t)
{
    clock_gettime(t->context.tc_posix->clock, &(t->context.tc_posix->ts_start));
}

void stopwatch_stop_posix(Stopwatch *t)
{
    clock_gettime(t->context.tc_posix->clock, &(t->context.tc_posix->ts_end));
}

double stopwatch_runtime_posix(Stopwatch *t)
{
    return (double) (t->context.tc_posix->ts_end.tv_sec
                            - t->context.tc_posix->ts_start.tv_sec)
            + (double) (t->context.tc_posix->ts_end.tv_nsec
                            - t->context.tc_posix->ts_start.tv_nsec) / 1e9;
}

void stopwatch_destroy_posix(Stopwatch *t)
{
    free(t->context.tc_posix);
    free(t);
}
