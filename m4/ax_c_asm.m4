dnl AX_C_ASM
dnl
dnl Confirm that the C compiler accepts one of the common inline assembly
dnl keywords, and save the result to $ac_cv_c_asm.
dnl
dnl Although the C standard has since C99 acknowledged that `asm` is a common
dnl extension for producing inline assembly, it is not a requirement.  We first
dnl test for `asm`, followed by common compiler-specific alternatives.
dnl
dnl This macro does not actually confirm that the keyword produces assembly
dnl output.  It only confirms that the result is a valid keyword.
dnl
dnl Perhaps there's some way to confirm the assembly output, such as with a 
dnl second argument similar to AC_CHECK_LIB, but I'm a bit reluctant to try
dnl that right now.
dnl
AC_DEFUN([AX_C_ASM], [
  AC_LANG_PUSH([C])
  AC_MSG_CHECKING([for $CC inline assembly keyword])
  AC_CACHE_VAL([ac_cv_c_asm], [
    ac_cv_c_asm="unknown"
    ac_asm_keywords="asm __asm__ __asm"
    for ac_asm_kw in $ac_asm_keywords; do
      AC_LINK_IFELSE(
        [AC_LANG_PROGRAM([], [$ac_asm_kw("");])],
        [ac_cv_c_asm=$ac_asm_kw]
      )
      AS_IF([test "$ac_cv_c_asm" != unknown], [break])
    done
  ])
  AS_CASE([$ac_cv_c_asm],
    [asm], [AC_MSG_RESULT([$ac_cv_c_asm])],
    [unknown], [AC_MSG_RESULT([unsupported])],
    [
      AC_DEFINE_UNQUOTED([asm],[$ac_cv_c_asm])
      AC_MSG_RESULT([$ac_cv_c_asm])
    ]
  )
  AC_LANG_POP([C])
])
