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

where :math:`f` is the maximum CPU frequency, :math:`N_\text{ops}` is the
maximum number of concurrent operations per cycle, and :math:`N_\text{vec}` is
the size of the vector registers.

For the Sandy Bridge architecture used on Raijin, the maximum turbo-boosted
frequency is a function of the number of active cores, and obeys the following
formula:

.. math::

   f = 2600 MHz + \left(4 + \left\lfloor \frac{8 - n}{2} \right\rfloor \right)
         \times 100 MHz

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

``y[i] = a * y[i]``
-------------------

Scalar-vector multiplication is shown in the code block below.

.. code:: c

   float y[N];
   float a;

   for (int i = 0; i < N; i++)
       y[i] = a * y[i];

For each iteration, there is one 4-byte load and one FLOP, so that the
arithmetic intensity is :math:`\frac{1}{4}`.  Based on our roofline diagram,
this operation is bounded by single-core peak performance of 26.4 GFLOP/sec.

(TODO: Explain performance as a function of vector size)

The observed peak performance is slightly below 12.8 GFLOP/sec, or nearly half
of peak.  This can be understood from the Intel-optimised assembly instructions
shown below.

.. code:: asm

   ..B2.6:
           vmulps    (%r14,%rdx,4), %ymm4, %ymm2
           vmulps    32(%r14,%rdx,4), %ymm4, %ymm3
           vmovups   %ymm2, (%r14,%rdx,4)
           vmovups   %ymm3, 32(%r14,%rdx,4)
           addq      $16, %rdx
           cmpq      %rdi, %rdx
           jb        ..B2.7


The loop has one extra unroll, and there are 10 micro-ops in this block: two
FLOPs and two moves, four memory offset calculations, and two loop counter
instructions.

Since the Sandy Bridge can only decode up to four instructions per cycle, this
loop requires at least three cycles.  So the best performance we can expect is
2 FLOPs per 3 cycles.

There are two ``vmulps`` multiplication instructions and the Sandy Bridge has
one AVX multiplication port, so these must be distributed over two cycles.
Each of these ``vmulps`` instructions also requires a load from memory, and
each AVX load requires two cycles, or one half of an AVX register (16 bytes)
per cycle.  But since there are two load ports, these loads can be similarly
staggered, so that the loads and FLOPs can be executed over two cycles.  The
code block is therefore not bounded by memory loads.

However, the code block is bounded by its memory stores.  Sandy Bridge only has
a single port dedicated to L1 memory writes, and each AVX write to memory
requires two cycles.  So the two memory writes of the ``movups`` instructions
require four cycles to execute, and our peak performance is 2 FLOPs per 4
cycles.

Therefore, the peak performance on our architecture is 13.2 GFLOP/sec, and we
observe ~96% efficiency on Raijin.

This simple example illustrates how we must consider multiple factors in a
roofline analysis.  In this case, there were three limiting factors:

* Load arithmetic intensity
* Store arithmetic intensity
* Micro-op decoding

The load and store arithmetic intensity for this case are both
:math:`\frac{1}{4}`, but the different L1 load and store speeds (32 and 16
bytes per cycle, respectively) result in different peak performances at
:math:`\frac{1}{4}` intensity, where loads are computationally bound but stores
are memory-bound.


``y[i] = y[i] + y[i]``
----------------------

A similar example is the addition of a vector with itself, as in the following
code block.

.. code:: c

   float y[N];
   float a;

   for (int i = 0; i < N; i++)
       y[i] = y[i] + y[i];

Again, the arithmetic load and store intensities are :math:`\frac{1}{4}`,
since there is one FLOP, one 4-byte read of ``y[i]``, and one 4-byte write back
to ``y[i]``.  Roofline analysis predicts a peak performance of 13.2 GFLOP/sec,
based on the L1 store bandwidth, and the observed performance is slightly below
12.8 GFLOP/sec.

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

Although there are more instructions, the addition instructions ``vaddps``
operate on an independent port from the loads and stores, so the behaviour is
otherwise identical to the first example.  The two loads to populate
``ymm0`` and ``ymm3`` can be done in two cycles using the two load ports, but
the single store port means that four cycles are required to transfer the
results from ``ymm2`` and ``ymm4`` to L1 memory.  Therefore, the two FLOPs
require four cycles to complete, yielding the 50% peak performance result.


``y[i] = x[i] + y[i]``
----------------------

Addition of two independent vectors introduces an additional layer of
complexity, although the net result is similar.  The example code block is
shown below.

.. code:: c

   float x[N], y[N];

   for (int i = 0; i < N; i++)
       y[i] = x[i] + y[i];

This time, each FLOP requires that we load two 4-byte floats, and the
arithmetic load intensity is :math:`\frac{1}{8}`.  Only one 4-byte float is
stores in memory, so the arithmetic store intensity if :math:`\frac{1}{4}`.





``y[i] = a * x[i] + y[i]``
--------------------------

