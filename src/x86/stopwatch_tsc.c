#include <stdint.h>     /* uint64_t */
#include <stdlib.h>     /* malloc, free */
#include <time.h>       /* clock[id]_* */
#include <stdio.h>

#include "stopwatch.h"

/* Header */
static double stopwatch_get_tsc_freq(void);

void stopwatch_init_tsc(Stopwatch *t)
{
    t->context.tc_tsc = malloc(sizeof(struct stopwatch_context_tsc_t));
    t->context.tc_tsc->cpufreq = stopwatch_get_tsc_freq();
}

void stopwatch_start_tsc(Stopwatch *t)
{
    /* CPUID ensures that instructions prior to RDTSC have been completed.
     *
     * Based on Gabriele Paolini's benchmark document for Intel.
     */

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
    /* RDTSCP serialises the RDTSC instruction, ensuring that instructions
     * prior to RDTSCP have completed.
     * Instructions after RDTSCP could be executed after the final instruction
     * but prior to RDTSCP.  This is prevented by the final CPUID instruction.
     *
     * Based on Gabriele Paolini's benchmark document for Intel.
     */

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
    /* A stripped-down rdtsc call, without the out-of-order calls or explicit
     * MOV instructions.  The bit shift is probably redundant and adding a few
     * cycles, but is not my main problem at the moment.
     */

    uint64_t rax, rdx;
    uint32_t aux;

    __asm__ __volatile__ ("rdtscp" : "=a" (rax), "=d" (rdx), "=c" (aux));
    return (rdx << 32) | rax;
}

double stopwatch_get_tsc_freq(void)
{
    /* This program attempts to determine the TSC frequency by using POSIX
     * timers and TSC counter readings.
     *
     * This is a volatile task, and seems to be generally discouraged by kernel
     * developers, but I am a daredevil.  The current implementation seems to produce
     * consistent results within about 0.1 kHz.
     *
     * The general method is to measure the time and the TSC counter before and
     * after a sufficiently "long" operation.  Longer operations will yield
     * higher precisions.
     *
     * This "long" operation is a loop of clock_gettime calls whose results are
     * ignored.  We use clock_gettime to ensure that the function and its call
     * stack remain in cache for the final time measurement.
     *
     * In order to determine the number of clock_gettime iterations in the
     * "long" step, we do a pre-calculation where the number of iterations is
     * doubled until the runtime exceeds some threshold.  Currently, this is
     * set to 0.1 seconds.
     *
     * The clock_gettime call is regarded as "slow" while rdtsc is "fast".  In
     * practice, the ratio of clock_gettime to rdtsc runtime can be anywhere
     * from 2 to 20 when in cache.  (Note: clock_gettime calls that are not in
     * cache can be nearly a thousand times slower than rdtsc calls;
     * eliminating this overhead is the motivation for using TSC counters.)
     *
     * Since rdtsc is "fast", we call this before and after the clock_gettime
     * measurements, and use the mean value to determine the effective TSC
     * counter during the clock_gettime call.  (Note: It does not seem
     * necessary to take the mean, either TSC measurement would be sufficient.
     * Need to look into this more.)
     *
     * This is all new to me, and as mentioned it's a volatile task, so there
     * is surely more to learn here.
     */

    uint64_t cycle_start1, cycle_start2;
    uint64_t cycle_end1, cycle_end2;
    struct timespec ts_start, ts_end;

    uint64_t cycles, d_start, d_end;
    double runtime;

    unsigned long ncalls;   /* 32 or 64 byte? */

    int verbose = 1;    /* Not yet supported */

    /* Determine the number of iterations to get ns precision */
    ncalls = 1;
    do {
        ncalls *= 2;

        clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);

        for (long i = 0; i < ncalls; i++)
            clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);

        clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);

        runtime = (double) (ts_end.tv_sec - ts_start.tv_sec)
                  + (double) (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

    } while (runtime < 0.1);

    /* Use ncalls to estimate the TSC frequency */
    do {
        /* "Warm the cache" with multiple clock_gettime calls. */
        for (long i = 0; i < ncalls; i++)
            clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);

        /* Match the first timestamp to TSC counters */
        cycle_start1 = rdtsc();
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
        cycle_start2 = rdtsc();

        /* Perform a "long" (~1 sec) calculation.
         * Use clock_gettime to keep the final TSC-metered call in cache. */
        for (long i = 0; i < ncalls; i++)
            clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);

        /* Match the final timestamp to the TSC counter */
        cycle_end1 = rdtsc();
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);
        cycle_end2 = rdtsc();

        runtime = (double) (ts_end.tv_sec - ts_start.tv_sec)
                  + (double) (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

        /* Use the mean rdtsc value before and after clock_gettime calls */
        cycles = ((cycle_end1 + cycle_end2)
                    - (cycle_start1 + cycle_start2)) / 2;

        /* Estimate the clock_gettime call time (assuming TSC is faster) */
        d_start = cycle_start2 - cycle_start1;
        d_end = cycle_end2 - cycle_end1;

        /* Diagnostic testing */
        if (verbose) {
            printf("Cycles: %lu\n", cycles);
            printf("Runtime: %.12f\n", runtime);
            printf("dstart: %lu\n", d_start);
            printf("dend: %lu\n", d_end);
            printf("TSC frequency: %.12f GHz\n",
                   (double) cycles / runtime / 1e9);
            printf("TSC residual: %.12f kHz\n",
                   1e6 * ((double) cycles / runtime / 1e9 - 1.79592));
        }

    } while (d_start / d_end > 2 || d_end / d_start > 2);

    return (double) cycles / runtime;
}
