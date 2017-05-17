#include <stdint.h>     /* uint64_t */
#include <stdlib.h>     /* malloc */
#include <time.h>

#include "stopwatch.h"

/* Global function tables and struct sizes */

const size_t stopwatch_context_size[TIMER_MAX] = {
    sizeof(stopwatch_context_posix_t),
    sizeof(stopwatch_context_tsc_t),
};

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

/* Generic Stopwatch methods */

Stopwatch * stopwatch_create(TimerType type)
{
    Stopwatch *t;

    t = malloc(sizeof(Stopwatch));
    t->context.tc_untyped = malloc(stopwatch_context_size[type]);

    t->start = stopwatch_start_funcs[type];
    t->stop = stopwatch_stop_funcs[type];
    t->runtime = stopwatch_runtime_funcs[type];
    t->destroy = stopwatch_destroy_funcs[type];

    stopwatch_init_funcs[type](t);

    return t;
}


/* TSC Stopwatch methods */

void stopwatch_init_tsc(Stopwatch *t)
{
    t->context.tc_tsc = malloc(sizeof(stopwatch_context_tsc_t));
    t->context.tc_tsc->cpufreq = stopwatch_get_tsc_freq(t);
}

void stopwatch_start_tsc(Stopwatch *t)
{
    __asm__ __volatile__ (
        "cpuid\n"
        "rdtsc\n"
        "movq %%rax, %0\n"
        "movq %%rdx, %1\n"
        : "=r" (t->context.tc_tsc->rax0), "=r" (t->context.tc_tsc->rdx0)
        :: "%rax", "%rbx", "%rcx", "%rdx"
    );
}

void stopwatch_stop_tsc(Stopwatch *t)
{
    __asm__ __volatile__ (
        "cpuid\n"
        "rdtsc\n"
        "movq %%rax, %0\n"
        "movq %%rdx, %1\n"
        : "=r" (t->context.tc_tsc->rax1), "=r" (t->context.tc_tsc->rdx1)
        :: "%rax", "%rbx", "%rcx", "%rdx"
    );
}

double stopwatch_runtime_tsc(Stopwatch *t)
{
    uint64_t t0, t1;

    t0 = (t->context.tc_tsc->rdx0 << 32) | t->context.tc_tsc->rax0;
    t1 = (t->context.tc_tsc->rdx1 << 32) | t->context.tc_tsc->rax1;

    return (t1 - t0) / t->context.tc_tsc->cpufreq;
}

void stopwatch_destroy_tsc(Stopwatch *t)
{
    free(t->context.tc_tsc);
}

/* TSC support functions */

double stopwatch_get_tsc_freq(Stopwatch *t)
{
    uint64_t t0, t1;
    struct timespec req;
    req.tv_sec = 1;
    req.tv_nsec = 0;

    t->start(t);
    nanosleep(&req, NULL);
    t->stop(t);

    t0 = (t->context.tc_tsc->rdx0 << 32) | t->context.tc_tsc->rax0;
    t1 = (t->context.tc_tsc->rdx1 << 32) | t->context.tc_tsc->rax1;

    return (double) (t1 - t0);
}

/* POSIX Stopwatch methods */

void stopwatch_init_posix(Stopwatch *t)
{
    t->context.tc_posix = malloc(sizeof(stopwatch_context_posix_t));
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
}
