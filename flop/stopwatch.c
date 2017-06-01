#include <stdint.h>     /* uint64_t */
#include <stdlib.h>     /* malloc, free */
#include <time.h>       /* clock[id]_* */
#include <stdio.h>

#include "stopwatch.h"

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

struct stopwatch_context_tsc_t {
    double cpufreq;
    uint64_t rax0, rdx0, rax1, rdx1;
};

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
    t->context.tc_untyped = malloc(stopwatch_context_size[type]);

    t->start = stopwatch_start_funcs[type];
    t->stop = stopwatch_stop_funcs[type];
    t->runtime = stopwatch_runtime_funcs[type];
    t->destroy = stopwatch_destroy_funcs[type];

    stopwatch_init_funcs[type](t);

    return t;
}

/* TSC methods */

void stopwatch_init_tsc(Stopwatch *t)
{
    t->context.tc_tsc = malloc(sizeof(struct stopwatch_context_tsc_t));
    t->context.tc_tsc->cpufreq = stopwatch_get_tsc_freq();
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
        "rdtscp\n"
        "movq %%rax, %0\n"
        "movq %%rdx, %1\n"
        "cpuid\n"
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

uint64_t rdtsc(void)
{
    uint64_t rax, rdx;
    uint32_t aux;

    __asm__ __volatile__ ( "rdtscp" : "=a" ( rax ), "=d" ( rdx ), "=c" (aux));

    return (rdx << 32) | rax;
}

double stopwatch_get_tsc_freq(void)
{
    uint64_t cycle_start1, cycle_start2;
    uint64_t cycle_end1, cycle_end2;
    struct timespec ts_start, ts_end, ts_sleep, ts_remain;

    uint64_t cycles;
    double runtime;

    int rt;

    /* Set the timer */
    ts_sleep.tv_sec = 1;
    ts_sleep.tv_nsec = 0;

    cycle_start1 = rdtsc();
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
    cycle_start2 = rdtsc();

    //rt = nanosleep(&ts_sleep, &ts_remain);
    rt = clock_nanosleep(CLOCK_MONOTONIC, 0, &ts_sleep, &ts_remain);
    //rt = usleep(1000000);

    cycle_end1 = rdtsc();
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);
    cycle_end2 = rdtsc();

    runtime = (double) (ts_end.tv_sec - ts_start.tv_sec)
              + (double) (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

    cycles = ((cycle_end1 + cycle_end2) - (cycle_start1 + cycle_start2)) / 2;

    //printf("Cycles: %llu\n", cycles);
    //printf("Runtime: %.12f\n", runtime);
    //printf("dstart: %llu\n", cycle_start2 - cycle_start1);
    //printf("dend: %llu\n", cycle_end2 - cycle_end1);
    //printf("TSC frequency: %.12f GHz\n", (double) cycles / runtime / 1e9);

    return (double) cycles / runtime;
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
}
