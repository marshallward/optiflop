#define _GNU_SOURCE     /* CPU_COUNT */

#include <getopt.h>     /* getopt_long, option */
#include <sched.h>      /* sched_getaffinity, cpu_set_t */
#include <stdio.h>      /* printf */
#include <stdlib.h>     /* abort */
#include <string.h>     /* strtok */

#include "input.h"

void parse_input(int argc, char *argv[], struct input_config *cfg)
{
    int optflag;
    int option_index;
    char *token;

    cpu_set_t cpuset;
    int nprocs;

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

    struct option long_options[] =
    {
        {"help", no_argument, &(cfg->print_help), 1},
        {"verbose", no_argument, &(cfg->verbose), 1},
        {"output", no_argument, 0, 'o'},
        {0, 0, 0, 0}
    };

    option_index = 0;
    while (1) {
        optflag = getopt_long(argc, argv, "ol:s:p:r:e:",
                              long_options, &option_index);

        if (optflag == -1)
            break;

        switch (optflag) {
            case 0:
                /* TODO */
                break;
            case 'e':
                cfg->ensembles = (int) strtol(optarg, (char **) NULL, 10);
            case 'o':
                cfg->save_output = 1;
                break;
            case 'l':
                token = strtok(optarg, ",");
                cfg->vlen_start = (int) strtol(token, (char **) NULL, 10);
                token = strtok(NULL, ",");
                if (token != NULL)
                    cfg->vlen_end = (int) strtol(token, (char **) NULL, 10);
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
        printf("The FLOPS benchmark\n");
        printf("\n");
        printf("Flags:\n");
        printf("    -l start[,stop] Test vectors lengths from `start` to `stop`\n");
        printf("    -s scale        Scale step ratio during vector sweep\n");
        printf("    -p N            Benchmark N processors\n");
        printf("    -o              Output results to `results.txt`\n");
        printf("    -e N            Number of ensembles\n");
        printf("    --verbose       Display per-thread performance\n");
        exit(EXIT_SUCCESS);
    }

    if (cfg->vlen_end < 0) cfg->vlen_end = cfg->vlen_start + 1;

    sched_getaffinity(0, sizeof(cpuset), &cpuset);
    nprocs = CPU_COUNT(&cpuset);

    if (cfg->nthreads > nprocs) {
        printf("flop: Number of threads (%i) exceeds maximum "
               "core count (%i).\n", cfg->nthreads, nprocs);
        exit(EXIT_FAILURE);
    }
}
