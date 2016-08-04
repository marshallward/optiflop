=================
Roofline examples
=================

We present a hierarchy of calculations with different arithmetic intensities,
and compare their performance with the upper bound based on roofline modelling

Peak performance
================

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

The peak performance, in GFLOPS per second, is summarised on the following
table:

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
