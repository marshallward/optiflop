#ifndef _GETTIME_H_
#define _GETTIME_H_


struct POSIX_TimeContext {
    clockid_t clock;
    struct timespec ts_start, ts_end;
};


struct RDTSC_TimeContext {
    uint64_t rax0, rdx0, rax1, rdx1;
    double cpufreq;
};

/* TODO: Move these into the struct... */

void posix_starttime(struct POSIX_TimeContext *time);
void posix_endtime(struct POSIX_TimeContext *time);
double posix_runtime(struct POSIX_TimeContext *time);
void rdtsc_starttime(struct RDTSC_TimeContext *time);
void rdtsc_endtime(struct RDTSC_TimeContext *time);
double rdtsc_runtime(struct RDTSC_TimeContext *time);

#endif
