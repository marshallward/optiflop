#ifndef TASK_H_
#define TASK_H_

/* Backend selection */
enum task_backend {
    TASK_UNDEF = -1,
    TASK_SERIAL,        /* Serial implementation */
    TASK_POSIX,         /* POSIX threads */
    TASK_MAX,
};


/* Context */
union task_context_t {
    struct task_context_serial_t *tc_serial;
    struct task_context_posix_t *tc_posix;
};


/* Task class */
typedef struct Task_struct {
    union task_context_t *context;
    /* TODO: API */
} Task;


/* Public Task methods */
Task * task_create(enum task_backend);

#endif  // TASK_H_
