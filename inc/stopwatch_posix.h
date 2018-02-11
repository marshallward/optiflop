#ifndef STOPWATCH_POSIX_H_
#define STOPWATCH_POSIX_H_

/* POSIX Timer methods */
void stopwatch_init_posix(Stopwatch *t);
void stopwatch_start_posix(Stopwatch *t);
void stopwatch_stop_posix(Stopwatch *t);
double stopwatch_runtime_posix(Stopwatch *t);
void stopwatch_destroy_posix(Stopwatch *t);

#endif  // STOPWATCH_POSIX_H_
