AC_INIT([optiflop], [0.1],
        [https://github.com/marshallward/optiflop/issues],
        [],
        [https://github.com/marshallward/optiflop.git])

AC_CONFIG_HEADERS([inc/config.h])
AC_CONFIG_MACRO_DIR([m4])


# Set CFLAGS to a default value to replace the AC_PROG_CC default (-g -O2)
: ${CFLAGS="-g -O3"}

# C compiler configuration
AC_LANG([C])
AC_PROG_CC


# Validate the following keywords
AX_C_ASM
AC_C_RESTRICT
AC_C_VOLATILE


# Enable built-in subroutine substitution, e.g. memcpy()
AC_MSG_CHECKING([whether built-in functions are to be used])
AC_ARG_ENABLE(builtin,
    AS_HELP_STRING(--enable-builtin, [enable built-in functions]))
if test "$enable_builtin" = no; then
    CFLAGS="$CFLAGS -fno-builtin"
else
    enable_builtin=yes
fi
AC_MSG_RESULT($enable_builtin)


# Enable direct streaming from RAM to cache via movnt
AC_MSG_CHECKING([whether streaming stores are to be used])
AC_ARG_ENABLE(stream,
    AS_HELP_STRING(--enable-stream, [enable streaming stores (e.g. STREAM)]))
if test "$ac_cc" = "icc" -a "$enable_stream" = yes; then
    CFLAGS="$CFLAGS -mcmodel=medium -ffreestanding -qopt-streaming-stores always"
else
    enable_stream=no
fi
AC_MSG_RESULT($enable_stream)


# Loop unroll check
AC_MSG_CHECKING([loop unroll factor])
AC_ARG_ENABLE(unroll,
    AS_HELP_STRING(--enable-unroll, [set the loop unrolling factor]))
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


# SIMD support
AC_DEFUN([AX_CHECK_FLAG],
[
    AC_MSG_CHECKING([if $CC supports $1])
    ac_saved_cflags="$CFLAGS"
    CFLAGS="-O0 -m$1"
    # SSE mm_set intrinsics do not include the word length in their prefix
    if test $3 -eq 128; then
        ac_mm_prefix=""
    else
        ac_mm_prefix="$3"
    fi
    AC_RUN_IFELSE(
        [AC_LANG_PROGRAM([#include <immintrin.h>],
                         [[const __m$3 r = _mm${ac_mm_prefix}_set1_ps(1.f);]]
        )],
        [$2=1]
        [AC_MSG_RESULT([yes])],
        [$2=0]
        [AC_MSG_RESULT([no])],
    )
    CFLAGS="$ac_saved_cflags"
])


# SIMD FMA intrinsic support
AC_DEFUN([AX_CHECK_FMA],
[
    AC_MSG_CHECKING([if $CC supports $1 FMA])
    ac_saved_cflags="$CFLAGS"
    CFLAGS="-O0 -m$1 -mfma"
    # SSE mm_set intrinsics do not include the word length in their prefix
    if test $3 -eq 128; then
        ac_mm_prefix=""
    else
        ac_mm_prefix="$3"
    fi
    AC_COMPILE_IFELSE(
        [AC_LANG_PROGRAM(
            [#include <immintrin.h>],
            [[__m$3 r;
              r = _mm${ac_mm_prefix}_fmadd_ps(r, r, r);
            ]]
        )],
        [$2=1]
        [AC_MSG_RESULT([yes])],
        [$2=0]
        [AC_MSG_RESULT([no])],
    )
    CFLAGS="$ac_saved_cflags"
])


# Reset cc_has_fma flag
cc_has_fma=0

AX_CHECK_FLAG([sse], cc_has_sse, 128)
if test $cc_has_sse -eq 1; then
    sse_path=x86/sse
    CFLAGS="$CFLAGS -msse"
    if test "$ac_cc" = "icc"; then
        CFLAGS="$CFLAGS -xsse"
    fi
    AX_CHECK_FMA([sse], cc_has_sse_fma, 256)
    if test "$cc_has_sse_fma" -eq 1; then
        cc_has_fma=1
        sse_fma_path=x86/sse
    else
        sse_fma_path=generic
    fi
else
    sse_path=generic
    sse_fma_path=generic
fi

AX_CHECK_FLAG([avx], cc_has_avx, 256)
if test $cc_has_avx -eq 1; then
    avx_path=x86/avx
    CFLAGS="$CFLAGS -mavx"
    if test "$ac_cc" = "icc"; then
        CFLAGS="$CFLAGS -xavx"
    fi
    AX_CHECK_FMA([avx], cc_has_avx_fma, 256)
    if test "$cc_has_avx_fma" -eq 1; then
        cc_has_fma=1
        avx_fma_path=x86/avx
    else
        avx_fma_path=generic
    fi
else
    avx_path=generic
    avx_fma_path=generic
fi

AX_CHECK_FLAG([avx512f], cc_has_avx512, 512)
if test $cc_has_avx512 -eq 1; then
    avx512_path=x86/avx512
else
    avx512_path=generic
fi

# FMA flag support
if test "$cc_has_fma" -eq 1; then
    CFLAGS="$CFLAGS -mfma"
fi

# x86 Vectorisation flags (need to redo this)
if test $cc_has_avx512 -eq 1 -a "$ac_cc" = "gcc"; then
    CFLAGS="$CFLAGS -mavx512f"
fi

if test $cc_has_avx512 -eq 1 -a "$ac_cc" = "icc"; then
    CFLAGS="$CFLAGS -mCORE-AVX512 -DBYTEALIGN=64"
fi


# x86 assembly check
AC_DEFUN([AX_CHECK_ASM], [
  AC_MSG_CHECKING([if $CC supports $1])
  ac_saved_cflags="$CFLAGS"
  CFLAGS="-O0"
  AC_RUN_IFELSE(
    [AC_LANG_PROGRAM([], [[$ac_cv_c_asm volatile ("$1");]])],
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
AX_CHECK_ASM(rdpmc, has_rdpmc)
AX_CHECK_ASM(cpuid, has_cpuid)
if test $has_rdtsc -eq 1 -a $has_rdtscp -eq 1 -a $has_cpuid -eq 1; then
    tsc_path=x86
else
    tsc_path=generic
fi


# Update and create the Makefile
AC_SUBST(loop_cflags, $loop_cflags)
AC_SUBST(sse_path, "$sse_path")
AC_SUBST(sse_fma_path, "$sse_fma_path")
AC_SUBST(avx_path, "$avx_path")
AC_SUBST(avx_fma_path, "$avx_fma_path")
AC_SUBST(avx512_path, "$avx512_path")
AC_SUBST(tsc_path, "$tsc_path")
AC_CONFIG_FILES([Makefile])
AC_OUTPUT
