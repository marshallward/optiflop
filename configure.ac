AC_INIT([microbench], [0.1],
        [https://github.com/marshallward/microbench/issues],
        [],
        [https://github.com/marshallward/microbench.git])

# Set CFLAGS to an empty default value to prevent AC_PROG_CC default (-g -O2)
: ${CFLAGS="-g -O3"}

AC_LANG(C)
AC_PROG_CC_STDC

# Control flags
AC_ARG_ENABLE(builtin,
    AC_HELP_STRING(--enable-builtin, [enable built-in functions])
])

if test "$enable_builtin" = "no"; then
    CFLAGS="$CFLAGS -fno-builtin"
fi

# Libraries
AC_SEARCH_LIBS(ceil, m)
AC_SEARCH_LIBS(clock_gettime, rt)

# Vectorisation flag check
AC_DEFUN([AX_CHECK_FLAG],
[
    AC_MSG_CHECKING([if $CC supports $1])
    ac_saved_cflags="$CFLAGS"
    CFLAGS=-m$1
    AC_RUN_IFELSE(
        [AC_LANG_PROGRAM([#include <immintrin.h>],
                         [[const __m$3 r = _mm$3_set1_ps(1.f);]]
        )],
        [$2=1]
        [ac_saved_cflags="$ac_saved_cflags $CFLAGS"]
        [AC_MSG_RESULT([yes])],
        [$2=0]
        [AC_MSG_RESULT([no])],
    )
    CFLAGS="$ac_saved_cflags"
])

AX_CHECK_FLAG([avx], cc_has_avx, 256)
if test $cc_has_avx -eq 1; then
    avx_path=x86
else
    avx_path=generic
fi

AX_CHECK_FLAG([avx512f], cc_has_avx512, 512)
if test $cc_has_avx512 -eq 1; then
    avx512_path=x86/avx512
else
    avx512_path=generic
fi

# x86 assembly check
AC_DEFUN([AX_CHECK_ASM],
[
    AC_MSG_CHECKING([if $CC supports $1])
    AC_RUN_IFELSE(
        [AC_LANG_PROGRAM([], [[__asm__ __volatile__ ("$1");]])],
        [$2=1]
        [AC_MSG_RESULT([yes])],
        [$2=0]
        [AC_MSG_RESULT([no])],
    )
])

# TODO: More checks are probably needed here (and some are probably redundant)
AX_CHECK_ASM(rdtsc, has_rdtsc)
AX_CHECK_ASM(rdtscp, has_rdtscp)
AX_CHECK_ASM(cpuid, has_cpuid)
if test $has_rdtsc -eq 1 -a $has_rdtscp -eq 1 -a $has_cpuid -eq 1; then
    tsc_path=x86
else
    tsc_path=generic
fi

# Update and create the Makefile
AC_SUBST(avx_path, "$avx_path")
AC_SUBST(avx512_path, "$avx512_path")
AC_SUBST(tsc_path, "$tsc_path")
AC_OUTPUT(Makefile)
