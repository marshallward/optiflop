void ROOF_TEST(int n, float a, float b,
               float * restrict x_in, float * restrict y_in,
               struct roof_args *args)
{
    float *x, *y;

    Stopwatch *t;

    long r, r_max;
    int i, offset;
    double runtime;

    offset = OFFSET_INDEX;

    x = ASSUME_ALIGNED(x_in);
    y = ASSUME_ALIGNED(y_in);

    t = stopwatch_create(TIMER_POSIX);

    r_max = 1;
    runtime_flag = 0;
    do {
        pthread_barrier_wait(&timer_barrier);
        t->start(t);
        for (r = 0; r < r_max; r++) {
            for (i = offset; i < n; i++)
                ROOF_KERNEL;
            // Create an impossible branch to prevent loop interchange
            if (y[0] < 0.) dummy(a, b, x, y);
        }
        t->stop(t);
        runtime = t->runtime(t);

        /* Set runtime flag if any thread exceeds runtime limit */
        if (runtime > (args->min_runtime)) {
            pthread_mutex_lock(&runtime_mutex);
            runtime_flag = 1;
            pthread_mutex_unlock(&runtime_mutex);
        }

        pthread_barrier_wait(&timer_barrier);
        if (!runtime_flag) r_max *= 2;

    } while (!runtime_flag);

    args->runtime = runtime;
    args->flops = ROOF_FLOPS * (n - offset) * r_max / runtime;
    args->bw_load = ROOF_BW_LOAD * (n - offset) * sizeof(float) * r_max
                        / runtime;
    args->bw_store = ROOF_BW_STORE * (n - offset) * sizeof(float) * r_max
                        / runtime;

    /* Cleanup */
    t->destroy(t);
}

#undef ROOF_TEST
#undef ROOF_KERNEL
#undef ROOF_FLOPS
#undef ROOF_BW_LOAD
#undef ROOF_BW_STORE
#ifdef ROOF_OFFSET
    #undef ROOF_OFFSET
#endif
