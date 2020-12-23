#ifndef STOPWATCH_STD_H_
#define STOPWATCH_STD_H_

/* C89 defines CLK_TCK rather than CLOCKS_PER_SEC */
#ifndef CLOCKS_PER_SEC
#define CLOCKS_PER_SEC CLK_TCK
#endif

/* C standard library implmentation of Stopwatch */
void stopwatch_init_std(Stopwatch *t);
void stopwatch_start_std(Stopwatch *t);
void stopwatch_stop_std(Stopwatch *t);
double stopwatch_runtime_std(Stopwatch *t);
void stopwatch_destroy_std(Stopwatch *t);

#endif  // STOPWATCH_STD_H_
