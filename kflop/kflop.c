#include <linux/module.h>       /* Linux module */
#include <linux/init.h>         /* entry/exit macros */
#include <linux/kernel.h>       /* printk */
#include <linux/hardirq.h>      /* raw_local_irq_save|restore */
#include <linux/preempt.h>      /* preempt_enable|disable */
#include <linux/sched.h>        /* task_struct */
#include <asm/fpu/api.h>        /* kernel_fpu_begin|end */

static int __init hello_start(void)
{
    unsigned long irq_flags;
    unsigned long rax0, rdx0, rax1, rdx1;
    unsigned long tsc0, tsc1;

    const unsigned long N = 10000000;
    unsigned long i;

    printk(KERN_INFO "Loading KernFlops.\n");

    preempt_disable();                  /* Disable preemption of CPU */
    raw_local_irq_save(irq_flags);      /* Disable hard interrupts on CPU */
    kernel_fpu_begin();                 /* Enable floating point */

    __asm__ __volatile__ (
        "cpuid\n"
        "rdtsc\n"
        "movq %%rax, %0\n"
        "movq %%rdx, %1\n"
        : "=r" (rax0), "=r" (rdx0)
        :: "%rax", "%rbx", "%rcx", "%rdx");

    /* Begin test */

    for (i = 0; i < N; i++) {
        __asm__ __volatile__ (
            "vaddps %ymm3, %ymm4, %ymm3\n"
            "vaddps %ymm2, %ymm4, %ymm2\n"
            "vaddps %ymm1, %ymm4, %ymm1\n"
            "vaddps %ymm0, %ymm4, %ymm0\n"
            "vsubps %ymm4, %ymm3, %ymm3\n"
            "vsubps %ymm4, %ymm2, %ymm2\n"
            "vsubps %ymm4, %ymm1, %ymm1\n"
            "vsubps %ymm4, %ymm0, %ymm0\n");
    }

    /* End test */

    __asm__ __volatile__ (
        "rdtscp\n"
        "movq %%rax, %0\n"
        "movq %%rdx, %1\n"
        "cpuid\n"
        : "=r" (rax1), "=r" (rdx1)
        :: "%rax", "%rbx", "%rcx", "%rdx" );

    tsc0 = (rdx0 << 32) | rax0;
    tsc1 = (rdx1 << 32) | rax1;

    printk(KERN_INFO "TSC count: %lu\n", tsc1 - tsc0);

    kernel_fpu_end();                   /* Disable floating point */
    raw_local_irq_restore(irq_flags);   /* Re-enable hard interrupts on CPU */
    preempt_enable();                   /* Re-enable preemption on CPU */

    return 0;
}

static void __exit hello_end(void)
{
    printk(KERN_INFO "Unloading KernFlops.\n");
}

module_init(hello_start);
module_exit(hello_end);

MODULE_AUTHOR("Marshall Ward");
MODULE_LICENSE("Dual BSD/GPL");
MODULE_DESCRIPTION("Kernel-space peak FLOP test.");
