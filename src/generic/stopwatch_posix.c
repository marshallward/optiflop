#include <stdio.h>      /* printf */
#include <stdlib.h>     /* exit */
#include "stopwatch.h"

void stopwatch_init_posix(Stopwatch *t)
{
    printf("ERROR: POSIX timer not supported.\n");
    exit(-1);
}

void stopwatch_start_posix(Stopwatch *t) {}
void stopwatch_stop_posix(Stopwatch *t) {}
double stopwatch_runtime_posix(Stopwatch *t) { return -1; }
void stopwatch_destroy_posix(Stopwatch *t) {}
