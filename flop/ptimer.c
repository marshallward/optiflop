#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>

struct Timer;
struct ITimer
{
    void (*start)(struct Timer *);
    void (*stop)(struct Timer *);
    double (*runtime)(struct Timer *);
};

struct Timer
{
    struct ITimer *vtable;
    /* base members */
};

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

struct TscTimer
{
    struct Timer super;

    double cpufreq;
    uint64_t rax0, rdx0, rax1, rdx1;
};

void timer_start_tsc(struct TscTimer *d)
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

void timer_stop_tsc(struct TscTimer *d)
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

double timer_runtime_tsc(struct TscTimer *d)
{
    uint64_t t0, t1;

    t0 = (d->rdx0 << 32) | d->rax0;
    t1 = (d->rdx1 << 32) | d->rax1;

    return (t1 - t0) / d->cpufreq;
}

/* global vtable for derived1 */
struct ITimer tsc_vtable =
{
    &timer_start_tsc, /* you might get a warning here about incompatible pointer types */
    &timer_stop_tsc,  /* you can ignore it, or perform a cast to get rid of it */
    &timer_runtime_tsc,
};

void timer_create_tsc(struct TscTimer *d)
{
    d->super.vtable = &tsc_vtable;
    d->cpufreq = 2.59e9;
}

/* POSIX Timer */

struct PosixTimer
{
    struct Timer super;

    clockid_t clock;
    struct timespec ts_start, ts_end;
};

void timer_start_posix(struct PosixTimer *d)
{
    clock_gettime(d->clock, &(d->ts_start));
}

void timer_stop_posix(struct PosixTimer *d)
{
    clock_gettime(d->clock, &(d->ts_end));
}

double timer_runtime_posix(struct PosixTimer *d)
{
    return (double) (d->ts_end.tv_sec - d->ts_start.tv_sec)
            + (double) (d->ts_end.tv_nsec - d->ts_start.tv_nsec) / 1e9;
}


struct ITimer posix_vtable =
{
    &timer_start_posix, /* you might get a warning here about incompatible pointer types */
    &timer_stop_posix,  /* you can ignore it, or perform a cast to get rid of it */
    &timer_runtime_posix,
};

void timer_create_posix(struct PosixTimer *d)
{
    d->super.vtable = &posix_vtable;
    d->clock = CLOCK_MONOTONIC_RAW;
}

int main(void)
{
    struct TscTimer d1;
    timer_create_tsc(&d1);

    struct PosixTimer d2;
    timer_create_posix(&d2);

    struct Timer *b1_ptr = (struct Timer *)&d1;
    struct Timer *b2_ptr = (struct Timer *)&d2;

    timer_start(b1_ptr);  /* calls derived1_dance */
    timer_start(b2_ptr);  /* calls derived2_dance */

    sleep(3);

    timer_stop(b1_ptr);  /* calls derived1_jump */
    timer_stop(b2_ptr);  /* calls derived2_jump */

    printf("TSC Runtime: %f\n", timer_runtime(b1_ptr));
    printf("POSIX Runtime: %f\n", timer_runtime(b2_ptr));

    return 0;
}
