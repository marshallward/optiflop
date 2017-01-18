#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>

/* Generic Timer */

struct Timer;
struct ITimer
{
    void (*start)(struct Timer *);
    void (*stop)(struct Timer *);
    double (*runtime)(struct Timer *);
};

struct Timer
{
    struct ITimer *vtable;
};

void timer_start(struct Timer *b);
void timer_stop(struct Timer *b);
double timer_runtime(struct Timer *b);

/* TSC Timer */

typedef struct _TscTimer
{
    struct Timer super;

    double cpufreq;
    uint64_t rax0, rdx0, rax1, rdx1;
} TscTimer;

void timer_start_tsc(TscTimer *d);
void timer_stop_tsc(TscTimer *d);
double timer_runtime_tsc(TscTimer *d);

struct ITimer tsc_vtable =
{
    &timer_start_tsc, /* you might get a warning here about incompatible pointer types */
    &timer_stop_tsc,  /* you can ignore it, or perform a cast to get rid of it */
    &timer_runtime_tsc,
};

void timer_create_tsc(TscTimer *d);

/* POSIX Timer */

typedef struct _PosixTimer
{
    struct Timer super;

    clockid_t clock;
    struct timespec ts_start, ts_end;
} PosixTimer;

void timer_start_posix(PosixTimer *d);
void timer_stop_posix(PosixTimer *d);
double timer_runtime_posix(PosixTimer *d);

struct ITimer posix_vtable =
{
    &timer_start_posix, /* you might get a warning here about incompatible pointer types */
    &timer_stop_posix,  /* you can ignore it, or perform a cast to get rid of it */
    &timer_runtime_posix,
};

void timer_create_posix(PosixTimer *d);
