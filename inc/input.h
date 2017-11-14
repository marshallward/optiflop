#ifndef ARGS_H_
#define ARGS_H_

#include "stopwatch.h"

struct input_config {
    int print_help;
    int verbose;
    int save_output;
    int vlen_start;
    int vlen_end;
    double vlen_scale;
    int nthreads;
    double min_runtime;
    int ensembles;
    enum stopwatch_backend timer_type;
};

void parse_input(int, char **, struct input_config *);

#endif  // ARGS_H_
