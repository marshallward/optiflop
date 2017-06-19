#!/usr/bin/env python
import csv
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np

# Theoretical performance
# (Sandy Bridge Raijin numbers below)

# TODO: Use avx_add/avx_mac to get these values
platform = 'Raijin, E5-2670'

n_avx = 8
f_peak = 3.3e9
P_peak = n_avx * f_peak

p_min, p_max = -6, 2
perf_grid = P_peak * 2.**np.arange(p_min, p_max + 1)

# Single-channel DRAM bandwidth
# (This is not exactly correct, fix it up...)
f_dram = 800e6      # RAM frequency (800 MHz)
ddr = 2             # Two sends per cycle (DDR) (DDR3 is more like 4 or 8?)
bus_width = 8       # 64-bit bus width
n_channels = 4      # Raijin has 4-channel DRAM (but don't use it here)
bw_dram = f_dram * ddr * bus_width  # Single-channel? (No...)

# L1 bandwidth
bw_l1l = 32 * f_peak    # 32-byte/cycle loads
bw_l1s = 16 * f_peak    # 16-byte/cycle stores

ai_l1l = P_peak / bw_l1l
ai_l1s = P_peak / bw_l1s
ai_dram = P_peak / bw_dram

# Read results

results = {}
with open('results.txt', 'r') as flopfile:

    flopreader = csv.reader(flopfile)
    for row in flopreader:
        n = int(row[0])
        try:
            results[n] = [float(r) for r in row[1:]]
        except:
            print(row)
            raise

x_ai = []
y_results = []
for i in range(len(results[1])):
    y = np.array([results[n][i] for n in results])
    y_results.append(y)
    x_ai.append(np.ones(y.size))

# Manually set the arithmetic intensities (because I'm a dummy)
x_ai[0][:] = 0.25 * x_ai[0][:]
x_ai[1][:] = 0.125 * x_ai[1][:]
x_ai[2][:] = 0.25 * x_ai[2][:]
x_ai[3][:] = 0.375 * x_ai[3][:]

# Plot results

fig, ax = plt.subplots()

ax.set_title('Single-core roofline ({})'.format(platform))
ax.set_xlabel('Arithmetic Intensity (FLOPs / byte)')
ax.set_ylabel('Performance (GFLOPs / sec)')

ai_min, ai_max = -6, 2
ai_grid = 2.**np.arange(ai_min, ai_max + 1)
ai_axis = np.array([Fraction(x).limit_denominator() for x in ai_grid])

ax.set_xscale('log')
ax.set_xlim(2.**ai_min, 2.**ai_max)
ax.set_xticks(ai_axis)
ax.set_xticklabels(ai_axis)

ax.set_yscale('log')
ax.set_ylim(P_peak * 2.**(p_min - 0.5), P_peak * 2.**p_max)
ax.set_yticks(perf_grid)
ax.set_yticklabels(['{:.1f}'.format(tk / 1e9) for tk in perf_grid])

ax.minorticks_off()

# Serialised peak
ax.axhline(P_peak / n_avx, color='k', linestyle=':')
ax.text(1., P_peak / n_avx * 1.1, 'Peak Serial')

# Single-port single-precision AVX peak
ax.plot([ai_min, ai_l1l], [P_peak, P_peak],
        color='k', linestyle=':')
ax.plot([ai_l1l, ai_max], [P_peak, P_peak],
        color='k', linestyle='-')
ax.text(1. / 48., P_peak * 1.1, 'Peak AVX (SP)')

# Concurrent single-precision add/multiply peak
ax.plot([ai_min, 2. * ai_l1l], [2. * P_peak, 2. * P_peak],
        color='k', linestyle=':')
ax.plot([2. * ai_l1l, ai_max], [2. * P_peak, 2. * P_peak],
        color='k', linestyle='-')
ax.text(1. / 48., 2. * P_peak * 1.1, 'Peak Add/Mult (SP)')

# L1 load bound
tx = 1. / 16.
ax.plot([ai_min, ai_l1l], [bw_l1l * ai_min, bw_l1l * ai_l1l],
        color='r', linestyle='-')
ax.plot([ai_l1l, 2. * ai_l1l], [bw_l1l * ai_l1l, bw_l1l * 2. * ai_l1l],
        color='r', linestyle='--')
ax.plot([2. * ai_l1l, ai_max], [bw_l1l * 2. * ai_l1l, bw_l1l * ai_max],
        color='r', linestyle=':')
ax.text(tx, bw_l1l * tx * 1.15,
        'L1 Load ({:.1f} GB/sec)'.format(bw_l1l / 1e9),
        rotation=45., ha='center', va='center')

# L1 store bound
tx = 1. / 8.
ax.plot([ai_min, ai_l1s], [bw_l1s * ai_min, bw_l1s * ai_l1s],
        color='b', linestyle='-')
ax.plot([ai_l1s, 2. * ai_l1s], [bw_l1s * ai_l1s, bw_l1s * 2. * ai_l1s],
        color='b', linestyle='--')
ax.plot([2. * ai_l1s, ai_max], [bw_l1s * 2. * ai_l1s, bw_l1s * ai_max],
        color='b', linestyle=':')
ax.text(tx, bw_l1s * tx * 1.15,
        'L1 Store ({:.1f} GB/sec)'.format(bw_l1s / 1e9),
        rotation=45., ha='center', va='center')

# DRAM bound
tx = 1. / 2.
ax.plot([ai_min, ai_dram], [bw_dram * ai_min, bw_dram * ai_dram],
        color='g', linestyle='-')
ax.plot([ai_dram, 2. * ai_dram], [bw_dram * ai_dram, bw_dram * 2. * ai_dram],
        color='g', linestyle='--')
ax.plot([2. * ai_dram, ai_max], [bw_dram * 2. * ai_dram, bw_dram * ai_max],
        color='g', linestyle=':')
ax.text(tx, bw_dram * tx * 1.15,
        'DRAM ({:.1f} GB/sec)'.format(bw_dram / 1e9),
        rotation=45., ha='center', va='center')

for (x, y) in zip(x_ai[1:4], y_results[1:4]):
    ax.scatter(x, y)

# Dumb: redraw (0) to put it in front
ax.scatter(x_ai[0], y_results[0])

plt.show()
