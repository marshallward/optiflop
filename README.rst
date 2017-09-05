========
Optiflop
========

A tool for peak performance and roofline analysis.


About Optiflop
==============

Optiflop is a program for measuring the peak computational performance
(in FLOPs per second) and memory bandwidth on a compute node.  Tests are
written in C with a goal of portability.


Quick usage guide
=================

This is still a development branch, so the build process requires an extra step
or two.

1. Generate the ``configure`` script and ``Makefile``, then ``make``.::

      $ autoconf
      $ ./configure
      $ make

To run for a default vector size (3200)::

   ./optiflop

For further options, ``optiflop --help``.


Test overview
=============

All tests consist of repeated loops over operations in C.  Scalar tests use
vector intrinsic calls.  Vector tests are written in standard C.  Compiler
intrinsics are used to ensure alignment.

The following tests are included:

``avx_add``
   Evaluation of 256-bit AVX ``vadd`` instructions over a set of registers,
   specified by the ``VADD_LATENCY`` macro.  This was meant to be a single-port
   performance test on Sandy Bridge architectures, but newer CPUs may have
   multiple ALU ports.

``avx_mac``
   Evaluation of concurrent 256-bit AVX ``vadd`` and ``vmul`` instructions over
   separate registers, set by ``VADD_LATENCY`` and ``VMUL_LATENCY``.  This is
   intended to be a dual-port performance test on Sandy Bridge architectures.

``avx512_add``
``avx512_mac``
  AVX512 implementations of ``avx_add`` and ``avx_mac``.  Must be enabled in
  the Makefile.


Roofline tests (*TODO*)
-----------------------

``roof_copy``

``roof_ax``

``roof_xpy``

``roof_axpy``

``roof_axpby``
