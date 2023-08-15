AC_INIT([optiflop], [0.1],
        [https://github.com/marshallward/optiflop/issues],
        [],
        [https://github.com/marshallward/optiflop.git])
AC_CONFIG_HEADERS([inc/config.h])
AC_CONFIG_MACRO_DIR([m4])


# Compiler configuration
AC_LANG([C])
AC_PROG_CC


# Validate the following keywords
AX_C_ASM
AC_C_RESTRICT
AC_C_VOLATILE


#---
# Timer configuration

# NOTES:
# - Check for clockid_t in <time.h> to avoid a false positive in <sys/types.h>.
# - librt (POSIX1.b realtime library) may need to be explicitly linked.
AC_CHECK_TYPE([clockid_t],
  [
    posix_timer_path='posix'
    AC_SEARCH_LIBS([clock_gettime], [rt])
  ],
  [posix_timer_path='generic'],
  [[#include <time.h>]]
)


# TODO: Our TSC test uses the POSIX timer, but we could relax this.
AS_IF([test $posix_timer_path = "posix"], [
  AX_CHECK_ASM([rdtsc], [has_rdtsc])
  AX_CHECK_ASM([rdtscp], [has_rdtscp])
  AX_CHECK_ASM([cpuid], [has_cpuid])
  AS_IF([test $has_rdtsc -eq 1 -a $has_rdtscp -eq 1 -a $has_cpuid -eq 1],
    [tsc_timer_path=x86],
    [tsc_timer_path=generic]
  )],
  [tsc_timer_path=generic]
)


# Check if platform requires an explicit link to libm (math).
AC_SEARCH_LIBS(ceil, m)


# SIMD configuration
AX_CHECK_SIMD([sse], [__m128], [r = _mm_set1_ps(1.f)])
AX_CHECK_SIMD([sse_fma], [__m128], [r = _mm_fmadd_ps(r,r,r)])
AX_CHECK_SIMD([avx], [__m256], [r = _mm256_set1_ps(1.f)])
AX_CHECK_SIMD([avx_fma], [__m256], [r = _mm256_fmadd_ps(r,r,r)])
AX_CHECK_SIMD([avx512], [__m512], [r = _mm512_set1_ps(1.f)])

# Use 64-byte alignment if AVX512 is enabled
AS_VAR_IF([avx512_path], [x86],
  [AS_VAR_APPEND([CFLAGS], [" -DBYTEALIGN=64"])]
)

# Not yet implemented, but could be used to test direct transfers
AX_CHECK_SIMD([sse_movnt], [__m128],
              [float *p = malloc(16); _mm_stream_ps(p,r)])
AX_CHECK_SIMD([avx_movnt], [__m256],
              [float *p = malloc(32); _mm256_stream_ps(p,r)])
AX_CHECK_SIMD([avx512_movnt], [__m512],
              [float *p = malloc(64); _mm512_stream_ps(p,r)])


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


# BLAS configuration
# TODO: Gracefully handle the missing header case...
AC_SEARCH_LIBS([cblas_dgemm], [blas openblas], [
  AC_CHECK_HEADERS([cblas.h], [
    cblas_path="blas"
  ], [
    cblas_path="generic"
  ])
], [
  cblas_path="generic"
])


# GPU configuration
AC_ARG_ENABLE([cuda],
  AS_HELP_STRING([--enable-cuda], [Enable CUDA tests])
)
AS_IF([test "$enable_cuda" == "yes"],
  [
    AC_PATH_PROG([NVCC], [nvcc], [no])
    AS_IF([test $NVCC == "no"], [AC_MSG_ERROR([No CUDA compiler found])])
    LD=$NVCC
  ], [
    LD=$CC
    LDFLAGS+=" -pthread"
  ]
)

AC_ARG_ENABLE([cublas],
  AS_HELP_STRING([--enable-cublas], [Enable cuBLAS tests])
)
AS_IF([test "$enable_cublas" == "yes"],
  [
    AS_IF([test "$enable_cuda" == "no"],
      [AC_MSG_ERROR([cuBLAS enabled without CUDA])]
    )
    cublas_path="cuda"
    LIBS+="$LIBS -lcublas"
  ],[
    cublas_path="generic"
  ]
)

# Update and create the Makefile
AC_SUBST(posix_timer_path, "$posix_timer_path")
AC_SUBST(tsc_timer_path, "$tsc_timer_path")
AC_SUBST(loop_cflags, $loop_cflags)
AC_SUBST(sse_path, "$sse_path")
AC_SUBST(sse_fma_path, "$sse_fma_path")
AC_SUBST(avx_path, "$avx_path")
AC_SUBST(avx_fma_path, "$avx_fma_path")
AC_SUBST(avx512_path, "$avx512_path")
AC_SUBST(cblas_path, "$cblas_path")
AC_SUBST(enable_cuda, "$enable_cuda")
AC_SUBST(enable_cublas, "$enable_cublas")
AC_SUBST(LD, "$LD")
AC_CONFIG_FILES([Makefile])
AC_OUTPUT
