#define _GNU_SOURCE     /* CPU_COUNT */

#include <getopt.h>     /* getopt_long, option */
#include <sched.h>      /* sched_getaffinity, cpu_set_t */
#include <stdio.h>      /* printf */
#include <stdlib.h>     /* abort */
#include <string.h>     /* strtok */

#include "input.h"
#include "stopwatch.h"

void parse_input(int argc, char *argv[], struct input_config *cfg)
{
    int option_index;
    char *token;

    cpu_set_t cpuset;
    int nprocs;
    int use_tsc;

    /* Default values */
    cfg->print_help = 0;
    cfg->verbose = 0;
    cfg->save_output = 0;
    cfg->vlen_start = 3200;
    cfg->vlen_end = -1;
    cfg->vlen_scale = 2.;
    cfg->nthreads = 1;
    cfg->min_runtime = 1e-2;
    cfg->ensembles = 1;
    cfg->timer_type = TIMER_POSIX;

    struct option long_options[] =
    {
        {"help", no_argument, NULL, 'h'},
        {"output", no_argument, NULL, 'o'},
        {"verbose", no_argument, &(cfg->verbose), 1},
        {"tsc", no_argument, &use_tsc, 1},
        {0, 0, 0, 0}
    };

    use_tsc = 0;
    option_index = 0;
    while (1) {
        int optflag = getopt_long(argc, argv, "he:l:op:r:s:",
                                  long_options, &option_index);

        if (optflag == -1)
            break;

        switch (optflag) {
            case 0:
                /* TODO */
                break;
            case 'h':
                cfg->print_help = 1;
                break;
            case 'e':
                cfg->ensembles = (int) strtol(optarg, (char **) NULL, 10);
                break;
            case 'o':
                cfg->save_output = 1;
                break;
            case 'l':
                token = strtok(optarg, ",");
                cfg->vlen_start = strtol(token, (char **) NULL, 10);
                token = strtok(NULL, ",");
                if (token != NULL)
                    cfg->vlen_end = strtol(token, (char **) NULL, 10);
                break;
            case 's':
                cfg->vlen_scale = strtod(optarg, NULL);
                break;
            case 'p':
                cfg->nthreads = (int) strtol(optarg, (char **) NULL, 10);
                break;
            case 'r':
                cfg->min_runtime = strtod(optarg, NULL);
                break;
            default:
                abort();
        }
    }

    if (cfg->print_help) {
        printf("microbench\n");
        printf("\n");
        printf("Flags:\n");
        printf("    -l start[,stop] Test vectors lengths from `start` to `stop`\n");
        printf("    -s scale        Scale step ratio during vector sweep\n");
        printf("    -p N            Benchmark N processors\n");
        printf("    -o              Output results to `results.txt`\n");
        printf("    -e N            Number of ensembles\n");
        printf("    --verbose       Display per-thread performance\n");
        printf("    -h, --help      Display this help information\n");
        exit(EXIT_SUCCESS);
    }

    if (cfg->vlen_end < 0) cfg->vlen_end = cfg->vlen_start + 1;

    sched_getaffinity(0, sizeof(cpuset), &cpuset);
    nprocs = CPU_COUNT(&cpuset);

    if (use_tsc) cfg->timer_type = TIMER_TSC;

    if (cfg->nthreads > nprocs) {
        printf("flop: Number of threads (%i) exceeds maximum "
               "core count (%i).\n", cfg->nthreads, nprocs);
        exit(EXIT_FAILURE);
    }
}
