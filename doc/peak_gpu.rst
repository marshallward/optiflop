Peak performance on Tesla K80
=============================

These are some very rough notes on K80 performance.  Currently, typical vector
operations used in ocean models (i.e. hyperbolic PDE solvers with no matrices)
do not get anywhere close to peak performance on these chips.

Hardware
--------

A Tesla K80 card is broken down as follows:

- Each K80 card has 2 GK210 GPUs

- Each GK210 contains 13 SMX units

- Each SMX unit has the following:

  - 192 single-precision (SP) FPUs
  - 64 double-precision (DP) FPUs

The clock speed of the card is from 562 MHz (base) to 875 MHs (turbo).  Most
estimates seem to use the lower frequency, so this is probably the one used
when all registers are active.

Peak performance is therefore the following:

   P_SP = 2 GPU * 13 SMX * 192 FP ops / cycle * 562 cycles / sec
        = 2.8 TFLOP/s

   P_DP = 2 * 13 * 64 * 562
        = 0.94 TFLOP/s

If we apply FMA operations, so that we do 2 ops per FPU, then double the
result:

   P_SP = 5.6 TFLOP/s
   P_DP = 1.8 TFLOP/s

This reproduces most of the numbers that I have seen in the press releases.

(Others seem to substitute 562 MHz for 875 MHz; not sure how this all works,
but the analysis is the same.)

K80s are now rather old GPU hardware, but presumably this sort of analysis will
continue to later NVIDIA cards.


Roofline analysis
-----------------

Work here is not well-documented yet, but the basic result is that I get
nowhere near this result for a typical vector update.  Will update this later
(or incorporate into the main document).



Peak A100 Performance
=====================

Still gathering info here...

- 108 SM units
- 32 FP64 cores per SM
- Clock speed ~ 1410 MHz (boost)

Peak is 108 x 32 x 1.410 x 2 = 9.746 TFLOP/s

Tensor cores are still a mystery here.
