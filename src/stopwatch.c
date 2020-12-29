#include <stdint.h>     /* uint64_t */
#include <stdlib.h>     /* malloc, free */
#include <time.h>       /* clock[id]_* */
#include <stdio.h>

#include "stopwatch.h"
#include "stopwatch_std.h"
#include "stopwatch_posix.h"
#include "stopwatch_tsc.h"

/* Method lookup tables */

void (*stopwatch_init_funcs[TIMER_MAX])(Stopwatch *t) = {
    stopwatch_init_std,
    stopwatch_init_posix,
    stopwatch_init_tsc,
};

void (*stopwatch_start_funcs[TIMER_MAX])(Stopwatch *t) = {
    stopwatch_start_std,
    stopwatch_start_posix,
    stopwatch_start_tsc,
};

void (*stopwatch_stop_funcs[TIMER_MAX])(Stopwatch *t) = {
    stopwatch_stop_std,
    stopwatch_stop_posix,
    stopwatch_stop_tsc,
};

double (*stopwatch_runtime_funcs[TIMER_MAX])(Stopwatch *t) = {
    stopwatch_runtime_std,
    stopwatch_runtime_posix,
    stopwatch_runtime_tsc,
};

void (*stopwatch_destroy_funcs[TIMER_MAX])(Stopwatch *t) = {
    stopwatch_destroy_std,
    stopwatch_destroy_posix,
    stopwatch_destroy_tsc,
};

/* Generic Stopwatch methods */

Stopwatch * stopwatch_create(enum stopwatch_backend type)
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


/* C standard library implementation of Stopwatch
 *
 * This is a problematic timer for many reasons:
 *  - It measures process time, rather than wall time.
 *  - Precision is limited by CLOCKS_PER_SEC, which POSIX sets to 1us.
 *  - clock_t is an opaque type (int, double, etc), and may exhibit overflow.
 *
 * However, it is defined in the C standard (as far back as ANSI C) so it is a
 * good fallback option if no other timers are available.
 */

struct stopwatch_context_std {
    clock_t c_start, c_end;
};

void stopwatch_init_std(Stopwatch *t)
{
    t->context = malloc(sizeof(union stopwatch_context));
    t->context->tc_std = malloc(sizeof(struct stopwatch_context_std));
}

void stopwatch_start_std(Stopwatch *t)
{
    t->context->tc_std->c_start = clock();
}

void stopwatch_stop_std(Stopwatch *t)
{
    t->context->tc_std->c_end = clock();
}

double stopwatch_runtime_std(Stopwatch *t)
{
    clock_t t_start, t_end;

    t_start = t->context->tc_std->c_start;
    t_end = t->context->tc_std->c_end;
    return (double) (t_end - t_start) / CLOCKS_PER_SEC;
}

void stopwatch_destroy_std(Stopwatch *t)
{
    free(t->context->tc_std);
    free(t->context);
    free(t);
}
