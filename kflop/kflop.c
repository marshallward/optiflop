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
    unsigned long flops, flop_hi, flop_lo;

    const unsigned long N = 10000000;
    unsigned long i;

    printk(KERN_INFO "Loading KernFlops.\n");

    local_irq_save(irq_flags);          /* Disable interrupts */
    kernel_fpu_begin();                 /* Disable preempt, save FP state */

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

    flops = 8 * 8 * N * 1800000000 / (tsc1 - tsc0);
    flop_hi = flops / 1000000000;
    flop_lo = flops % 1000000000;

    printk(KERN_INFO "TSC count: %lu\n", tsc1 - tsc0);
    printk(KERN_INFO "FLOP/sec: %lu.%lu\n", flop_hi, flop_lo);

    kernel_fpu_end();                   /* Restore FP state, enable preempts */
    local_irq_restore(irq_flags);       /* Enable interrupts */

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
