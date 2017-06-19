#include <stdint.h>     /* uint64_t */
#include <stdlib.h>     /* malloc, free */
#include <time.h>       /* clock[id]_* */
#include <stdio.h>

#include "stopwatch.h"

void stopwatch_init_tsc(Stopwatch *t) 
{
    printf("ERROR: TSC timer not supported.\n");
    exit(-1);
}

void stopwatch_start_tsc(Stopwatch *t) {}
void stopwatch_stop_tsc(Stopwatch *t) {}
double stopwatch_runtime_tsc(Stopwatch *t) { return -1; }
void stopwatch_destroy_tsc(Stopwatch *t) {};
