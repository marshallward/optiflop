=================
Roofline examples
=================

We present a hierarchy of calculations with different arithmetic intensities,
and compare their performance with the upper bound based on roofline modelling

Peak performance
================

FLOPs
-----

The peak performance of the CPU is

.. math::

   P_\text{peak} = f \times N_\text{ops} \times N_\text{vec}

where :math:`f` is the maximum CPU frequency, :math:`N_\text{ops}` is the maximum
number of concurrent operations per cycle, and :math:`N_\text{vec}` is the
size of the vector registers.

For the Sandy Bridge architecture used on Raijin, the maximum turbo-boosted frequency
is a function of the number of active cores, and obeys the following formula:

.. math::

   f = 2600 MHz + \left(4 + \left\lfloor \frac{8 - n}{2} \right\rfloor \right) \times 100 MHz

so that :math:`f` is equal to 3.3 GHz when one core is active, and 3.0 GHz when
all 8 cores are active.  The non-turbo (TSC) frequency is 2.6 GHz.

Since the Sandy Bridge architecture supports the AVX instruction set, it
supports arithmetic over packed 256-bit (32-byte) registers.  For 4-byte
single-precision floating point numbers, we can compute over 8 values
simultaneously, so that :math:`N_\text{vec} = 8`.

The Sandy Bridge scheduler contains six independent ports for simultaneous
operations, including independent ports for addition and multiplication.  Under
optimal conditions, the Sandy Bridge is capable of concurrent AVX addition and
multiplication, enabling it compute two vectorised floating point operations
(FLOPS) over an AVX register per cycle, so that the peak value of
:math:`N_\text{ops}` is two.

Sandy Bridge also provides two separate ports for the simultaneous loading of
two values from the L1 cache, as well as additional arithmetic logic unit (ALU)
ports, shared with the addition and multiplication ports.

The peak performance, in GFLOPS per second, on Raijin is summarised on the
following table:

=====    ======   ======   ======   ======   ======   ======
Cores    SP Add   SP Mul   SP MAC   DP Add   DP Mul   DP MAC
=====    ======   ======   ======   ======   ======   ======
1          26.4     26.4     52.8     13.2     13.2     26.4
2          26.4     26.4     52.8     13.2     13.2     26.4
3          25.6     25.6     51.2     12.8     12.8     25.6
4          25.6     25.6     51.2     12.8     12.8     25.6
5          24.8     24.8     49.6     12.4     12.4     24.8
6          24.8     24.8     49.6     12.4     12.4     24.8
7          24.0     24.0     48.0     12.0     12.0     24.0
8          24.0     24.0     48.0     12.0     12.0     24.0
TSC        20.8     20.8     41.6     10.4     10.4     20.8
=====    ======   ======   ======   ======   ======   ======

- *SP*:  Single-precision
- *DP*:  Double-precision
- *Add*: Addition
- *Mul*: Multiplication
- *MAC*: Concurrent addition-multiplication
- *TSC*: Time Stamp Counter, referring to the non-turbo TSC frequency

.. TODO Haswell 12-core peak flops


Minimum peak arithmetic intensity
---------------------------------

Need to show that :math:`\frac{1}{4}` is the peak L1 arithmetic intensity.


Register arithmetic
===================

The most idealised example is repeated arithmetic on registers, where the
memory transfer is effectively zero and the arithmetic intensity is effectively
infinite.  In other words, the performance is compute-bound and is bounded by
the CPU's peak performance.

We present two cases relevant to the Sandy Bridge architecture: addition and
concurrent multiply/add operations.

Addition over registers
-----------------------

TODO (sort of started this already)


Concurrent addition-multiplication
----------------------------------

TODO


Vector arithmetic
=================

``y[i] = a y[i]``
-----------------

Scalar-vector multiplication is shown in the code block below:

.. code:: c

   float y[N];
   float a;

   for (int i = 0; i < N; i++)
       y[i] = a * y[i];

For each iteration, there is one 4-byte load and one FLOP, so that the
arithmetic intensity is :math:`\frac{1}{4}`.  Based on our roofline diagram,
this operation is bounded by single-core peak performance of 26.4 GFLOP/sec.

The observed peak performance is slightly below 12.8 GFLOP/sec, or nearly
half of peak.  This can be understood from the assembly instructions:

.. code:: asm

   ..B2.6:
           vmulps    (%r14,%rdx,4), %ymm4, %ymm2
           vmulps    32(%r14,%rdx,4), %ymm4, %ymm3
           vmovups   %ymm2, (%r14,%rdx,4)
           vmovups   %ymm3, 32(%r14,%rdx,4)
           addq      $16, %rdx
           cmpq      %rdi, %rdx
           jb        ..B2.7


There are 10 micro-ops in this loop: two FLOPs and two moves, each with two
memory load/stores, and two loop counter instructions.

Since the Sandy Bridge can only decode up to four instructions per cycle, this
loop is already bounded by at least three cycles.  So the best performance we
can expect is 2 FLOPs per 3 cycles.

Additionally, the memory load/stores require two cycles to complete for AVX
operations, processing 128 bits (16 bytes) per cycle, and there are only two
ports available.  So this ultimately leads to a stalling of the unrolled loop
on our architecture.  (This is wrong, it's the single-port p4 bottleneck.)

Therefore, the peak performance on our architecture is 13.2 GFLOP/sec, and we
observe ~96% efficiency on Raijin.

This example illustrates that Roofline performance predictions represent an
upper bound between CPU cycles and L1 bandwidth, and that additional
constraints may be present.


``y[i] = y[i] + y[i]``
----------------------

A similar example is the addition of a vector with itself, as in the following
code block.

.. code:: c

   float y[N];
   float a;

   for (int i = 0; i < N; i++)
       y[i] = y[i] + y[i];

Again, the peak roofline performance of this block is :math:`\frac{1}{4}`,
since there is one FLOP per 4-byte access, ``y[i]``, and peak performance is
26.4 GFLOP/s.  But again, the observed performance is slightly below 12.8
GFLOP/sec.

The assembly code shows a similar story to the ``y[i] = a * y[i]`` loop.

.. code:: asm

   ..B2.7:
           vmovups   (%r14,%rdx,4), %ymm0
           vmovups   32(%r14,%rdx,4), %ymm3
           vaddps    %ymm0, %ymm0, %ymm2
           vaddps    %ymm3, %ymm3, %ymm4
           vmovups   %ymm2, (%r14,%rdx,4)
           vmovups   %ymm4, 32(%r14,%rdx,4)
           addq      $16, %rdx
           cmpq      %rdi, %rdx
           jb        ..B2.7

For this code block with extra loop unroll, there are 12 micro-ops: 2 FLOPs, 4
moves, 4 memory load/stores, and 2 loop increments.  So the loop is again
bounded by 3 cycles and 2 FLOPs per 3 cycles.

The loop is further bound again by its number of loads.



``y[i] = x[i] + y[i]``
----------------------

``y[i] = a * x[i] + y[i]``
--------------------------

