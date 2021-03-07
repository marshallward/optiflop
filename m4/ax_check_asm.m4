dnl AX_CHECK_ASM
dnl
dnl Confirm that the compiler can support a given assembly function.
dnl Note that this goes through the C compiler.
AC_DEFUN([AX_CHECK_ASM], [
  AC_LANG_PUSH([C])
  AC_MSG_CHECKING([if CPU supports $1])
  ac_chk_asm_save_CFLAGS="$CFLAGS"
  CFLAGS="-O0"
  AC_RUN_IFELSE(
    [AC_LANG_PROGRAM([], [[$ac_cv_c_asm volatile ("$1");]])],
    [$2=1]
    [AC_MSG_RESULT([yes])],
    [$2=0]
    [AC_MSG_RESULT([no])],
  )
  CFLAGS="$ac_chk_asm_save_CFLAGS"
  AC_LANG_POP([C])
])
