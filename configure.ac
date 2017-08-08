AC_INIT([microbench], [0.1],
        [marshall.ward@gmail.com],
        [],
        [https://github.com/marshallward/microbench.git])

AC_LANG(C)
AC_PROG_CC

# x86 assembly check
AC_DEFUN([AX_CHECK_ASM],
[AC_RUN_IFELSE(
    [AC_LANG_PROGRAM([], [[__asm__ __volatile__ ($1);]])],
    [$2=1],
    [$2=0],
)
])

AX_CHECK_ASM("rdtsc", has_rdtsc)
AX_CHECK_ASM("rdtscp", has_rdtscp)
AX_CHECK_ASM("cpuid", has_cpuid)

AC_OUTPUT(Makefile)
