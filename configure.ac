AC_INIT([microbench], [0.1],
        [marshall.ward@gmail.com],
        [],
        [https://github.com/marshallward/microbench.git])

AC_LANG(C)
AC_PROG_CC
AC_PROG_CC_STDC

# Compiler flag check
AC_DEFUN([AX_CHECK_FLAG],
[
    AC_MSG_CHECKING([if $CC supports $1])
    ac_saved_cflags="$CFLAGS"
    CFLAGS=$1
    AC_COMPILE_IFELSE(
        [AC_LANG_PROGRAM([])],
        [$2=1]
        [AC_MSG_RESULT([yes])],
        [$2=0]
        [AC_MSG_RESULT([no])],
    )
    CFLAGS="$ac_saved_cflags"
])

AX_CHECK_FLAG([-mavx], cc_has_avx)

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

AC_SUBST(tsc_path, "$tsc_path")
AC_OUTPUT(Makefile)
