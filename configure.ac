AC_INIT([microbench], [0.1],
        [https://github.com/marshallward/microbench/issues],
        [],
        [https://github.com/marshallward/microbench.git])

# Set CFLAGS to a default value to replace the AC_PROG_CC default (-g -O2)
: ${CFLAGS="-g -O3"}

AC_LANG(C)
AC_PROG_CC_STDC

# NOTE: The STDC check appends -std=gnu99 to $CC
# Split compiler name and flags
ac_ifs=$IFS
IFS=" "
read ac_cc ac_cc_flags <<< $CC
IFS=$ac_ifs


# Enable built-in subroutine substitution, e.g. memcpy()
AC_MSG_CHECKING([whether built-in functions are to be used])
AC_ARG_ENABLE(builtin, [
    AC_HELP_STRING(--enable-builtin, [enable built-in functions])
])
if test "$enable_builtin" = no; then
    CFLAGS="$CFLAGS -fno-builtin"
else
    enable_builtin=yes
fi
AC_MSG_RESULT($enable_builtin)


# Enable direct streaming from RAM to cache via movnt
AC_MSG_CHECKING([whether streaming stores are to be used])
AC_ARG_ENABLE(stream,
    AC_HELP_STRING(--enable-stream, [enable streaming stores (e.g. STREAM)]))
if test "$ac_cc" = "icc" -a "$enable_stream" = yes; then
    CFLAGS="$CFLAGS -mcmodel=medium -ffreestanding -qopt-streaming-stores always"
else
    enable_stream=no
fi
AC_MSG_RESULT($enable_stream)


# Loop unroll check
AC_MSG_CHECKING([loop unroll factor])
AC_ARG_ENABLE(unroll,
    AC_HELP_STRING(--enable-unroll, [set the loop unrolling factor]))
if test "$enable_unroll" = "no"; then
    # Set default unroll rate to two
    enable_unroll=0
fi

if test "$enable_unroll" = "yes" -o "$enable_unroll" = ""; then
    enable_unroll=2
fi

# Confirm that unroll factor is a positive integer
case "$enable_unroll" in
    [[!0-9]]*)
        AC_MSG_ERROR([Loop unroll must be an integer]) ;;
    *)
        ;;
esac

if test $enable_unroll -gt 0; then
    loop_cflags="-funroll-loops --param max-unroll-times=$enable_unroll"
else
    loop_cflags=""
fi
AC_MSG_RESULT($enable_unroll)


# Libraries
AC_SEARCH_LIBS(ceil, m)
AC_SEARCH_LIBS(clock_gettime, rt)


# Vectorisation check
AC_DEFUN([AX_CHECK_FLAG],
[
    AC_MSG_CHECKING([if $CC supports $1])
    ac_saved_cflags="$CFLAGS"
    CFLAGS="-O0 -m$1"
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

AX_CHECK_FLAG([avx], cc_has_avx, 256)
if test $cc_has_avx -eq 1; then
    avx_path=x86
    CFLAGS="$CFLAGS -mavx"
else
    avx_path=generic
fi

AX_CHECK_FLAG([avx512f], cc_has_avx512, 512)
if test $cc_has_avx512 -eq 1; then
    avx512_path=x86/avx512
else
    avx512_path=generic
fi

# x86 Vectorisation flags (need to redo this)
if test $cc_has_avx512 -eq 1 -a "$ac_cc" = "gcc"; then
    CFLAGS="$CFLAGS -mavx512f"
fi

if test $cc_has_avx512 -eq 1 -a "$ac_cc" = "icc"; then
    CFLAGS="$CFLAGS -mCORE-AVX512 -DBYTEALIGN=64"
fi


# x86 assembly check
AC_DEFUN([AX_CHECK_ASM],
[
    AC_MSG_CHECKING([if $CC supports $1])
    ac_saved_cflags="$CFLAGS"
    CFLAGS="-O0"
    AC_RUN_IFELSE(
        [AC_LANG_PROGRAM([], [[__asm__ __volatile__ ("$1");]])],
        [$2=1]
        [AC_MSG_RESULT([yes])],
        [$2=0]
        [AC_MSG_RESULT([no])],
    )
    CFLAGS="$ac_saved_cflags"
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
AC_SUBST(loop_cflags, $loop_cflags)
AC_SUBST(avx_path, "$avx_path")
AC_SUBST(avx512_path, "$avx512_path")
AC_SUBST(tsc_path, "$tsc_path")
AC_OUTPUT(Makefile)
