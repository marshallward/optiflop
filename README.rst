=================
Raijin Benchmarks
=================

My collection of Raijin benchmarking programs.  Nearly everything here is based
on something that I've seen elsewhere.

FLOP/s counter
==============

These are based on Alexander Yee's Flops_ program.

avx_add
   Single-precision AVX addition FLOP test.  We add and subtract two
   nearly-equal numbers to the entries of a ``__m256`` vector (hopefully)
   stored in an AVX register.

avx_mac
   Single-precision synchronous add-and-multiply AVX FLOP test.  


.. _Flops: https://github.com/Mysticial/Flops
