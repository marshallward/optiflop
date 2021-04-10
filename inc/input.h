#ifndef OPTIFLOP_ARGS_H_
#define OPTIFLOP_ARGS_H_

#include "stopwatch.h"

struct input_config {
    int print_help;
    int verbose;
    int save_output;
    long vlen_start;
    long vlen_end;
    double vlen_scale;
    int nthreads;
    double min_runtime;
    int ensembles;
    enum stopwatch_backend timer_type;
};

void parse_input(int, char **, struct input_config *);

#endif  // OPTIFLOP_ARGS_H_
