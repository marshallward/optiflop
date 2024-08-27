AC_INIT([optiflop], [0.1],
        [https://github.com/marshallward/optiflop/issues],
        [],
        [https://github.com/marshallward/optiflop.git])
AC_CONFIG_HEADERS([inc/config.h])
AC_CONFIG_MACRO_DIR([m4])


# Compiler configuration
AC_LANG([C])
AC_PROG_CC

# Validate the following C keywords
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


# Threading
# TODO: Someday might generalize this, but for now we just force POSIX threads.
LDFLAGS="$LDFLAGS -pthread"


# Check if platform requires an explicit link to the math (libm) library.
AC_SEARCH_LIBS([ceil], [m])


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


# C BLAS configuration
AC_ARG_ENABLE([blas],
  AS_HELP_STRING([--enable-blas], [Enable BLAS testing])
)
# TODO: Gracefully handle the missing header case...
# TODO: Invert this loop!!
AS_IF([test "$enable_blas" == "yes"], [
  AC_SEARCH_LIBS([cblas_dgemm], [blas openblas], [
    AC_CHECK_HEADERS([openblas/cblas.h], [
      cblas_path="blas"
    ], [
      cblas_path="generic"
    ])
  ], [
    cblas_path="generic"
  ])
], [
  cblas_path="generic"
])


# GPU configuration
AC_ARG_ENABLE([cuda], [
  AS_HELP_STRING([--enable-cuda], [Enable CUDA tests])
])
# GPU (CUDA) BLAS configuration
AC_ARG_ENABLE([cublas], [
  AS_HELP_STRING([--enable-cublas], [Enable cuBLAS tests])
])


# If using CUDA, then use a C++ compatible compiler as linker.
# Otherwise, use the C compiler.
AS_IF([test "$enable_cuda" == "yes" -o "$enable_cublas" == "yes"], [
  AC_PATH_PROG([NVCC], [nvcc], [no])
  AS_IF([test $NVCC == "no"], [AC_MSG_ERROR([No CUDA compiler found])])

  # NOTE: g++ may be viable here, need more info
  AC_PATH_PROG([NVCPP], [nvc++], [no])
  AS_IF([test "$NVCPP" == "no"], [AC_MSG_ERROR([No C++ compiler found])])
  LD=$NVCPP
  # TODO: Test for libraries
  LIBS+="$LIBS -lcuda -lcudart"
], [
  LD=$CC
])


# cuBLAS setup
AS_IF([test "$enable_cublas" == "yes"], [
  cublas_path="cuda"
  LIBS+="$LIBS -lcublas"
],[
  cublas_path="generic"
])


# TODO: As Fortran tests are added, this criterion can be adjusted.
AS_IF([test "$enable_blas" == "yes"], [
  # NOTE: Test fixed format (.f) in a modern Fortran compiler (FC)
  #   In the future, we may need to test both fixed and free format.
  AC_LANG([Fortran])
  AC_FC_SRCEXT([f])
  AC_PROG_FC
  # NOTE: Older autoconfs seem to require a mature libtool setup.
  #   Not yet sure how to handle this with older versions (if at all).
  # NOTE: 2.69 definitely does not work, no idea about 2.70--2.71
  m4_version_prereq([2.72], [AC_FC_LIBRARY_LDFLAGS], [FCLIBS=""])
])


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
AC_SUBST(enable_blas, "$enable_blas")
AC_SUBST(enable_cuda, "$enable_cuda")
AC_SUBST(enable_cublas, "$enable_cublas")
AC_SUBST(LD, "$LD")
AC_SUBST(FCLIBS, "$FCLIBS")
AC_CONFIG_FILES([Makefile])
AC_OUTPUT
