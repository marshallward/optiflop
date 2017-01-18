#include <time.h> /* timespec, clock_gettime */
#include <stdint.h> /* uint64_t */
#include "gettime.h"

/* TODO: Merge the start and end functions? */

void posix_starttime(struct POSIX_TimeContext *time)
{
    clock_gettime(time->clock, &(time->ts_start));
}


void posix_endtime(struct POSIX_TimeContext *time)
{
    clock_gettime(time->clock, &(time->ts_end));
}


double posix_runtime(struct POSIX_TimeContext *time)
{
    return (double) (time->ts_end.tv_sec - time->ts_start.tv_sec)
            + (double) (time->ts_end.tv_nsec - time->ts_start.tv_nsec) / 1e9;
}


void rdtsc_starttime(struct RDTSC_TimeContext *time)
{
    __asm__ __volatile__ (
        "cpuid\n"
        "rdtsc\n"
        "movq %%rax, %0\n"
        "movq %%rdx, %1\n"
        : "=r" (time->rax0), "=r" (time->rdx0)
        :: "%rax", "%rbx", "%rcx", "%rdx"
    );
}


void rdtsc_endtime(struct RDTSC_TimeContext *time)
{
    __asm__ __volatile__ (
        "cpuid\n"
        "rdtsc\n"
        "movq %%rax, %0\n"
        "movq %%rdx, %1\n"
        : "=r" (time->rax1), "=r" (time->rdx1)
        :: "%rax", "%rbx", "%rcx", "%rdx"
    );
}


double rdtsc_runtime(struct RDTSC_TimeContext *time)
{
    uint64_t t0, t1;

    t0 = (time->rdx0 << 32) | time->rax0;
    t1 = (time->rdx1 << 32) | time->rax1;

    return (t1 - t0) / time->cpufreq;
}
