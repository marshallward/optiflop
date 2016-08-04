/* freq: estimate CPU frequency without reading proc
 *       This is cribbed from Score-P
 */

#include <stdint.h> /* uint64_t */
#include <stdio.h>  /* printf */
#include <time.h>   /* timespec, clock_gettime */

uint64_t tsc_freq(void);
uint64_t rdtsc(void);

int main(int argc, char *argv[])
{
    uint64_t ticks;

    ticks = tsc_freq();

    return 0;
}


uint64_t tsc_freq(void)
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

    printf("Cycles: %llu\n", cycles);
    printf("Runtime: %.12f\n", runtime);
    printf("dstart: %llu\n", cycle_start2 - cycle_start1);
    printf("dend: %llu\n", cycle_end2 - cycle_end1);
    printf("TSC frequency: %.12f GHz\n", (double) cycles / runtime / 1e9);

    return 0;
}


uint64_t rdtsc(void)
{
    uint64_t rax, rdx;
    uint32_t aux;

    __asm__ __volatile__ ( "rdtscp" : "=a" ( rax ), "=d" ( rdx ), "=c" (aux));

    return (rdx << 32) | rax;
}
