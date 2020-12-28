#include <malloc.h>

#include "task.h"
#include "task_serial.h"
#include "task_posix.h"

/* Method lookup tables */

void (*task_init_funcs[TASK_MAX])(Task *t) = {
    task_init_serial,
    task_init_posix,
};

/* Task API */

Task * task_create(enum task_backend type)
{
    Task *t;

    t = malloc(sizeof(Task));

    task_init_funcs[type](t);

    return t;
}

struct task_context_serial_t { };

void task_init_serial(Task *t)
{
    t->context = malloc(sizeof(union task_context_t));
    t->context->tc_serial = malloc(sizeof(struct task_context_serial_t));
}
