#include <getopt.h>     /* getopt_long, option */
#include <omp.h>        /* opt_get_num_procs */
#include <stdio.h>      /* printf */
#include <stdlib.h>     /* abort */

#include "input.h"

void parse_input(int argc, char *argv[], struct input_config *cfg)
{
    int optflag;
    int option_index;

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

    struct option long_options[] =
    {
        {"help", no_argument, &(cfg->print_help), 1},
        {"verbose", no_argument, &(cfg->verbose), 1},
        {"output", no_argument, 0, 'o'},
        {0, 0, 0, 0}
    };

    option_index = 0;
    while (1) {
        optflag = getopt_long(argc, argv, "ol:e:s:p:r:",
                              long_options, &option_index);

        if (optflag == -1)
            break;

        switch (optflag) {
            case 0:
                /* TODO */
                break;
            //case 'h':
            //    cfg->print_help = 1;
            //    break;
            case 'o':
                cfg->save_output = 1;
                break;
            case 'l':
                cfg->vlen_start = (int) strtol(optarg, (char **) NULL, 10);
                break;
            case 'e':
                cfg->vlen_end = (int) strtol(optarg, (char **) NULL, 10);
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
        printf("Help info here.\n");
        abort();    /* TODO: Don't abort here */
    }

    if (cfg->vlen_end < 0) cfg->vlen_end = cfg->vlen_start + 1;

    /* TODO: Get nproces without OpenMP (i.e. using affinity functions) */
    nprocs = omp_get_num_procs();
    if (cfg->nthreads > nprocs) {
        printf("flop: Number of threads (%i) exceeds maximum "
               "core count (%i).\n", cfg->nthreads, nprocs);
        abort();
    }
}
