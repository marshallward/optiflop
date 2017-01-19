#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>

#include "timer.h"

/* Generic Timer */

void timer_start(struct Timer *b)
{
    b->vtable->start(b);
}

void timer_stop(struct Timer *b)
{
    b->vtable->stop(b);
}

double timer_runtime(struct Timer *b)
{
    b->vtable->runtime(b);
}

/* TSC Timer */

void timer_start_tsc(TscTimer *d)
{
    __asm__ __volatile__ (
        "cpuid\n"
        "rdtsc\n"
        "movq %%rax, %0\n"
        "movq %%rdx, %1\n"
        : "=r" (d->rax0), "=r" (d->rdx0)
        :: "%rax", "%rbx", "%rcx", "%rdx"
    );
}

void timer_stop_tsc(TscTimer *d)
{
    __asm__ __volatile__ (
        "cpuid\n"
        "rdtsc\n"
        "movq %%rax, %0\n"
        "movq %%rdx, %1\n"
        : "=r" (d->rax1), "=r" (d->rdx1)
        :: "%rax", "%rbx", "%rcx", "%rdx"
    );
}

double timer_runtime_tsc(TscTimer *d)
{
    uint64_t t0, t1;

    t0 = (d->rdx0 << 32) | d->rax0;
    t1 = (d->rdx1 << 32) | d->rax1;

    return (t1 - t0) / d->cpufreq;
}

ITimer tsc_vtable =
{
    (timer_start_t) &timer_start_tsc,
    (timer_stop_t) &timer_stop_tsc,
    (timer_runtime_t) &timer_runtime_tsc,
};

void timer_create_tsc(TscTimer *d)
{
    d->super.vtable = &tsc_vtable;
    d->cpufreq = 2.601e9;
}

/* POSIX Timer */

void timer_start_posix(PosixTimer *d)
{
    clock_gettime(d->clock, &(d->ts_start));
}

void timer_stop_posix(PosixTimer *d)
{
    clock_gettime(d->clock, &(d->ts_end));
}

double timer_runtime_posix(PosixTimer *d)
{
    return (double) (d->ts_end.tv_sec - d->ts_start.tv_sec)
            + (double) (d->ts_end.tv_nsec - d->ts_start.tv_nsec) / 1e9;
}

ITimer posix_vtable =
{
    (timer_start_t) &timer_start_posix,
    (timer_stop_t) &timer_stop_posix,
    (timer_runtime_t) &timer_runtime_posix,
};

void timer_create_posix(PosixTimer *d)
{
    d->super.vtable = &posix_vtable;
    d->clock = CLOCK_MONOTONIC_RAW;
}
