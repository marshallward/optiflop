#include <malloc.h>
#include <pthread.h>

#include "task.h"
#include "task_posix.h"

struct task_context_posix_t {
    /* TODO: Presumably needed at some point...
    pthread_mutex_t mutex;
    pthread_attr_t attr;
    pthread_barrier_t barrier;
    */

    /* ??? */
    /* volatile int runtime_flag */
};

void task_init_posix(Task *tm)
{
    tm->context = malloc(sizeof(union task_context_t));
    tm->context->tc_posix = malloc(sizeof(struct task_context_posix_t)); 
}
