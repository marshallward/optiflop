dnl AX_CHECK_SIMD
dnl Check if Intel SIMD intrinsics are supported by the compiler.
dnl
dnl NOTE: These intrinsics are macros, not functions, and will often
dnl   only fail after executation of invalid assembly, so require execution.
dnl
AC_DEFUN([AX_CHECK_SIMD], [
  AC_LANG_PUSH([C])
  AC_MSG_CHECKING([if $CC supports $1])
  AC_RUN_IFELSE([
    AC_LANG_PROGRAM([#include <immintrin.h>],
                    [[volatile $2 r; $3;]])
  ], [
    AS_VAR_SET([$1_path], [x86])
    AC_MSG_RESULT([yes])
  ], [
    AS_VAR_SET([$1_path], [generic])
    AC_MSG_RESULT([no])
  ])
  AC_LANG_POP([C])
])
