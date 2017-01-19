#include <stdio.h>
#include <stdint.h>
#include <stdlib.h> /* malloc */
#include <time.h>
#include <unistd.h>

#include "timer.h"

/* Put this somewhere else... */

const size_t TimerContextSize[TIMER_MAX] = {
    sizeof(*((TimerContext *) NULL)->tc_posix),
    sizeof(*((TimerContext *) NULL)->tc_tsc),
};

/* Method lookup tables */
void (*timer_init_funcs[TIMER_MAX])(Timer *t) = {
    timer_init_posix,
    timer_init_tsc,
};

void (*timer_start_funcs[TIMER_MAX])(Timer *t) = {
    timer_start_posix,
    timer_start_tsc,
};

void (*timer_stop_funcs[TIMER_MAX])(Timer *t) = {
    timer_stop_posix,
    timer_stop_tsc,
};

double (*timer_runtime_funcs[TIMER_MAX])(Timer *t) = {
    timer_runtime_posix,
    timer_runtime_tsc,
};


/* Generic Timer methods */

Timer * mtimer_create(TimerType type)
{
    Timer *t;

    t = malloc(sizeof(Timer));
    t->context.tc_untyped = malloc(TimerContextSize[type]);

    timer_init_funcs[type](t);

    t->start = timer_start_funcs[type];
    t->stop = timer_stop_funcs[type];
    t->runtime = timer_runtime_funcs[type];

    return t;
}


/* TSC Timer */

void timer_init_tsc(Timer *t)
{
    t->context.tc_tsc = malloc(sizeof(TimerContextTsc));
    t->context.tc_tsc->cpufreq = 2.601e9;
}

void timer_start_tsc(Timer *t)
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

void timer_stop_tsc(Timer *t)
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

double timer_runtime_tsc(Timer *t)
{
    uint64_t t0, t1;

    t0 = (t->context.tc_tsc->rdx0 << 32) | t->context.tc_tsc->rax0;
    t1 = (t->context.tc_tsc->rdx1 << 32) | t->context.tc_tsc->rax1;

    return (t1 - t0) / t->context.tc_tsc->cpufreq;
}


/* POSIX Timer */

void timer_init_posix(Timer *t)
{
    t->context.tc_posix = malloc(sizeof(TimerContextPosix));
    t->context.tc_posix->clock = CLOCK_MONOTONIC_RAW;
}

void timer_start_posix(Timer *t)
{
    clock_gettime(t->context.tc_posix->clock, &(t->context.tc_posix->ts_start));
}

void timer_stop_posix(Timer *t)
{
    clock_gettime(t->context.tc_posix->clock, &(t->context.tc_posix->ts_end));
}

double timer_runtime_posix(Timer *t)
{
    return (double) (t->context.tc_posix->ts_end.tv_sec
                            - t->context.tc_posix->ts_start.tv_sec)
            + (double) (t->context.tc_posix->ts_end.tv_nsec
                            - t->context.tc_posix->ts_start.tv_nsec) / 1e9;
}

/*
void timer_create_posix(Timer *d)
{
    d->super.vtable = &posix_vtable;
    d->clock = CLOCK_MONOTONIC_RAW;
}
*/
