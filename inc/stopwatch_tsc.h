#ifndef STOPWATCH_TSC_H_
#define STOPWATCH_TSC_H_

/* Internal TSC Timer methods */
void stopwatch_init_tsc(Stopwatch *t);
void stopwatch_start_tsc(Stopwatch *t);
void stopwatch_stop_tsc(Stopwatch *t);
double stopwatch_runtime_tsc(Stopwatch *t);
void stopwatch_destroy_tsc(Stopwatch *t);

#endif  // STOPWATCH_TSC_H_
