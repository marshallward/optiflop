from fractions import Fraction
import math

import matplotlib.pyplot as plt
import numpy as np

def roofline():
    n_avx = 8   # AVX instruction set
    f_peak = cpufreq(1)
    P_peak = f_peak * n_avx

    ai_min, ai_max = -6, 2
    ai_grid = 2.**np.arange(ai_min, ai_max + 1)
    ai_axis = np.array([Fraction(x).limit_denominator() for x in ai_grid])

    p_min, p_max = -4, 2
    perf_grid = P_peak * 2.**np.arange(p_min, p_max + 1)

    fig, ax = plt.subplots()

    ax.set_xlabel('Arithmetic Intensity (FLOPs / byte)')
    ax.set_ylabel('Performance (FLOPs / sec)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(2.**ai_min, 2.**ai_max)
    ax.set_ylim(P_peak * 2.**(p_min - 0.5), P_peak * 2.**p_max)

    ax.set_xticks(ai_axis)
    ax.set_xticklabels(ai_axis)

    # Y-ticks relative to Peak FLOPs (in GFLOP/sec)
    ax.set_yticks(perf_grid)
    ax.set_yticklabels(['{:.1f}'.format(tk / 1e9) for tk in perf_grid])

    ax.minorticks_off()

    # L1 bandwidth
    bw_l1l = 32 * f_peak
    bw_l1s = 16 * f_peak

    # Single-channel DRAM bandwidth
    f_dram = 800e6      # RAM frequency (800 MHz)
    ddr = 2             # Two sends per cycle (DDR)
    bus_width = 8       # 64-bit bus width
    n_channels = 4      # Raijin has 4-channel DRAM (but don't use it here)

    bw_dram = f_dram * ddr * bus_width  # Single-channel!

    # Plot peak performance
    ai_min = np.min(ai_grid)
    ai_l1l = P_peak / bw_l1l
    ai_l1s = P_peak / bw_l1s
    ai_max = np.max(ai_grid)

    ax.axhline(P_peak / n_avx, color='k', linestyle=':')
    ax.text(2., P_peak / n_avx * 1.1, 'Serial')

    ax.plot([ai_min, ai_l1l], [P_peak, P_peak],
            color='k', linestyle=':')
    ax.plot([ai_l1l, ai_max], [P_peak, P_peak],
            color='k', linestyle='-')
    ax.text(1. / 32., P_peak * 1.1, 'SP AVX')

    ax.axhline(2. * P_peak, color='k', linestyle='--')
    ax.text(1. / 32., 2. * P_peak * 1.1, 'SP MAC')

    # L1 load bound
    tx = 1. / 16.
    ax.plot([ai_min, ai_l1l], [bw_l1l * ai_min, bw_l1l * ai_l1l],
            color='r', linestyle='-')
    ax.plot([ai_l1l, ai_max], [bw_l1l * ai_l1l, bw_l1l * ai_max],
            color='r', linestyle=':')
    ax.text(tx, bw_l1l * tx * 1.15,
            'L1 Load ({:.1f} GB/sec)'.format(bw_l1l / 1e9),
            rotation=45., ha='center', va='center')

    # L1 store bound
    tx = 1. / 8.
    ax.plot(ai_grid, bw_l1s * ai_grid, color='b', linestyle='--')
    ax.text(tx, bw_l1s * tx * 1.15,
            'L1 Store ({:.1f} GB/sec)'.format(bw_l1s / 1e9),
            rotation=45., ha='center', va='center')

    tx = 1. / 2.
    ax.plot(ai_grid, bw_dram * ai_grid, color='g', linestyle='--')
    ax.text(tx, bw_dram * tx * 1.15,
            'DRAM ({:.1f} GB/sec)'.format(bw_dram / 1e9),
            rotation=45., ha='center', va='center')


    plt.savefig('roofline.png')


def cpufreq(n):
    f_tsc = 2.601e9
    f_step = 0.1e9

    assert(0 < n <= 8)

    freq = f_tsc + f_step * (4 + math.floor((8. - n) / 2.))

    return freq


if __name__ == '__main__':
    roofline()
