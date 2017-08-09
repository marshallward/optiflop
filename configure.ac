AC_INIT([microbench], [0.1],
        [marshall.ward@gmail.com],
        [],
        [https://github.com/marshallward/microbench.git])

AC_LANG(C)
AC_PROG_CC
AC_PROG_CC_STDC

# Vectorisation flag check
AC_DEFUN([AX_CHECK_FLAG],
[
    AC_MSG_CHECKING([if $CC supports $1])
    ac_saved_cflags="$CFLAGS"
    CFLAGS=$1
    AC_RUN_IFELSE(
        [AC_LANG_PROGRAM([#include <immintrin.h>],
                         [[const __m$3 r = _mm$3_set1_ps(1.f);]]
        )],
        [$2=1]
        [AC_MSG_RESULT([yes])],
        [$2=0]
        [AC_MSG_RESULT([no])],
    )
    CFLAGS="$ac_saved_cflags"
])

AX_CHECK_FLAG([-mavx], cc_has_avx, 256)
if test $cc_has_avx -eq 1; then
    avx_path=x86
else
    avx_path=generic
fi

AX_CHECK_FLAG([-mavx512f], cc_has_avx512, 512)
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
if test $has_rdtsc -a $has_rdtscp -a $has_cpuid; then
    tsc_path=x86
else
    tsc_path=generic
fi

# Update and create the Makefile
AC_SUBST(avx_path, "$avx_path")
AC_SUBST(avx512_path, "$avx512_path")
AC_SUBST(tsc_path, "$tsc_path")
AC_OUTPUT(Makefile)
