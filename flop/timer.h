#ifndef _TIMER_H_
#define _TIMER_H_

#include <stdint.h> /* uint64_t */
#include <time.h> 	/* clockid_t, timespec */


/* Timer definition */

typedef enum _timer_t {
    TIMER_UNDEF = -1,
    TIMER_POSIX = 0,
    TIMER_TSC,
    TIMER_MAX,
} TimerType;


typedef struct _TimerContextTsc
{
    double cpufreq;
    uint64_t rax0, rdx0, rax1, rdx1;
} TimerContextTsc;


typedef struct _TimerContextPosix
{
    clockid_t clock;
    struct timespec ts_start, ts_end;
} TimerContextPosix;


typedef union _TimerContext {
    void *tc_untyped;
    TimerContextPosix *tc_posix;
    TimerContextTsc *tc_tsc;
} TimerContext;


typedef struct _Timer {
    TimerType type; /* Unneeded? */
    TimerContext context;

    void (*start)();
    void (*stop)();
    double (*runtime)();
} Timer;


/* Generic Timer methods */
Timer * mtimer_create(TimerType);

/* TSC Timer methods */

void timer_init_tsc(Timer *t);
void timer_start_tsc(Timer *t);
void timer_stop_tsc(Timer *t);
double timer_runtime_tsc(Timer *t);


/* POSIX Timer methods */

void timer_init_posix(Timer *t);
void timer_start_posix(Timer *t);
void timer_stop_posix(Timer *t);
double timer_runtime_posix(Timer *t);

#endif /* timer.h */
