#!/usr/bin/env python
import csv
from fractions import Fraction
import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np

# Config
outfile = 'gaea_roofline.svg'
text_color = None

platform = {
    'name': 'Gaea C4',
    'cpu': {
        'name': 'Intel E5-2697 v4',
        'freq': 3.6e9,
        'vlen': 4,          # Vector length (e.g. AVX)
        'l1_load': 64,      # L1 load bytes/cycle
        'l1_store': 32,     # L1 load bytes/cycle
    },
    'mem': {
        'freq': 1200e6,
    }
}

## Home machine
#platform {
#    'name': 'Home',
#    'cpu': {
#        'name': 'AMD Ryzen 5 2600',
#        'freq': 3.9e9,
#        'vlen': 4,
#        # TODO: No idea what these are!  Look them up...
#        'l1_load': 64,
#        'l1_store': 32,
#    },
#    'mem': {
#        'freq': 1066.67e6,
#    }
#}

# TODO: Use argparse
if len(sys.argv) == 2:
    results_fname = sys.argv[1]
else:
    results_fname = 'results.txt'

run_names = [
    "y[:] = a x[:]",
    "y[:] = x[:] + x[:]",
    "y[:] = x[:] + y[:]",
    "y[:] = a x[:] + y[:]",
    "y[:] = a x[:] + b y[:]",
    "y[1:] = x[1:] + x[:-1]",
    "y[8:] = x[8:] + x[:-8]",
    "GPU: y[:] = a * x[:] + y[:]",
]

# Theoretical performance
# TODO: Use avx_add/avx_mac to get these values

v_len = platform['cpu']['vlen']

f_peak = platform['cpu']['freq']
P_peak = v_len * f_peak

p_min_exp, p_max_exp = -6, 2
p_min, p_max = 2**p_min_exp, 2**p_max_exp
perf_grid = P_peak * 2.**np.arange(p_min_exp, p_max_exp + 1)

# Single-channel DRAM bandwidth

# Home machine (DDR4-1066)
#f_dram = 1066.67e6  # RAM frequency (1066 MHz)

# Gaea
f_dram = platform['mem']['freq']

ddr = 2             # Two sends per cycle (DDR) (DDR3 is more like 4 or 8?)
bus_width = 8       # 64-bit bus width
n_channels = 4      # Raijin has 4-channel DRAM (but don't use it here)
bw_dram = f_dram * ddr * bus_width  # Single-channel? (No...)

# L1 bandwidth
bw_l1l = platform['cpu']['l1_load'] * f_peak
bw_l1s = platform['cpu']['l1_store'] * f_peak

ai_l1l = P_peak / bw_l1l
ai_l1s = P_peak / bw_l1s
ai_dram = P_peak / bw_dram

# Read results

results = {}
with open(results_fname, 'r') as flopfile:

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
    # Temporarily fill AI with 1's
    x_ai.append(np.ones(y.size))

# .. then manually set the arithmetic intensities (because I'm a dummy)
x_ai[0][:] = 2**(-3.12)  # 1/8
x_ai[1][:] = 2**(-3.0)  # 1/8
x_ai[2][:] = 2**(-4.12)  # 1/16
x_ai[3][:] = 2**(-2.88)  # 1/8
x_ai[4][:] = 3./16.
x_ai[5][:] = 2**(-4.0)  # 1/16
x_ai[6][:] = 2**(-3.88)  # 1/16

# Plot results

fig, ax = plt.subplots(figsize=(6.,5.))

ax.set_title('Single-core roofline ({})'.format(platform['cpu']['name']))
ax.set_xlabel('Arithmetic Intensity (FLOPs / byte)')
ax.set_ylabel('Performance (GFLOPs / sec)')

ai_min_exp, ai_max_exp = -6, 2
ai_min, ai_max = 2**ai_min_exp, 2**ai_max_exp
ai_grid = 2.**np.arange(ai_min_exp, ai_max_exp + 1)
ai_axis = np.array([Fraction(x).limit_denominator() for x in ai_grid])

ax.set_xscale('log')
ax.set_xlim(ai_min, ai_max)
ax.set_xticks(ai_grid)
ax.set_xticklabels(ai_axis)

ax.set_yscale('log')
ax.set_ylim(P_peak * p_min / 2, P_peak * p_max)
ax.set_yticks(perf_grid)
ax.set_yticklabels(['{:.1f}'.format(tk / 1e9) for tk in perf_grid])

ax.minorticks_off()

# Serialised peak
ax.axhline(P_peak / v_len, color='k', linestyle=':')
ax.text(1., P_peak / v_len * 1.1, 'Peak Serial')

# Single-port AVX peak
ax.plot([ai_min, ai_l1l], [P_peak, P_peak],
        color='k', linestyle=':')
ax.plot([ai_l1l, ai_max], [P_peak, P_peak],
        color='k', linestyle='-')
ax.text(1. / 60., P_peak * 1.1, 'Peak AVX')

# Single-port FMA (or two-port add/multiply peak)
ax.plot([ai_min, 2. * ai_l1l], [2. * P_peak, 2. * P_peak],
        color='k', linestyle=':')
ax.plot([2. * ai_l1l, ai_max], [2. * P_peak, 2. * P_peak],
        color='k', linestyle='-')
ax.text(1. / 60., 2. * P_peak * 1.1, 'Peak FMA / 2x AVX')

# L1 load bound
tx = 1. / 45.
ax.plot([ai_min, ai_l1l], [bw_l1l * ai_min, bw_l1l * ai_l1l],
        color='r', linestyle='-')
ax.plot([ai_l1l, 2. * ai_l1l], [bw_l1l * ai_l1l, bw_l1l * 2. * ai_l1l],
        color='r', linestyle='--')
ax.plot([2. * ai_l1l, ai_max], [bw_l1l * 2. * ai_l1l, bw_l1l * ai_max],
        color='r', linestyle=':')
ax.text(tx, bw_l1l * tx * 1.15,
        #'L1 Load ({:.1f} GB/sec)'.format(bw_l1l / 1e9),
        'L1 Load'.format(bw_l1l / 1e9),
        rotation=35., ha='center', va='center')

# L1 store bound
tx = 1. / 45.
ax.plot([ai_min, ai_l1s], [bw_l1s * ai_min, bw_l1s * ai_l1s],
        color='b', linestyle='-')
ax.plot([ai_l1s, 2. * ai_l1s], [bw_l1s * ai_l1s, bw_l1s * 2. * ai_l1s],
        color='b', linestyle='--')
ax.plot([2. * ai_l1s, ai_max], [bw_l1s * 2. * ai_l1s, bw_l1s * ai_max],
        color='b', linestyle=':')
ax.text(tx, bw_l1s * tx * 1.15,
        #'L1 Store ({:.1f} GB/sec)'.format(bw_l1s / 1e9),
        'L1 Store'.format(bw_l1s / 1e9),
        rotation=35., ha='center', va='center')

# DRAM bound
tx = 1. / 45.
ax.plot([ai_min, ai_dram], [bw_dram * ai_min, bw_dram * ai_dram],
        color='g', linestyle='-')
ax.plot([ai_dram, 2. * ai_dram], [bw_dram * ai_dram, bw_dram * 2. * ai_dram],
        color='g', linestyle='--')
ax.plot([2. * ai_dram, ai_max], [bw_dram * 2. * ai_dram, bw_dram * ai_max],
        color='g', linestyle=':')
ax.text(tx, bw_dram * tx * 1.15,
        #'DRAM ({:.1f} GB/sec)'.format(bw_dram / 1e9),
        'DRAM'.format(bw_dram / 1e9),
        rotation=35., ha='center', va='center')

for (x, y, name) in zip(x_ai[:7], y_results[:7], run_names[:7]):
    ax.scatter(x, y, alpha=0.06, label=name)

leg = ax.legend()
for lh in leg.legendHandles:
    lh.set_alpha(1)

if text_color:
    ctxt = text_color
    ax.xaxis.label.set_color(ctxt)
    ax.yaxis.label.set_color(ctxt)
    ax.title.set_color(ctxt)
    for label in itertools.chain(ax.get_xticklabels(), ax.get_yticklabels()):
        label.set_color(ctxt)
    ax.tick_params(colors=ctxt, which='both')

fig.tight_layout()
if outfile:
    plt.savefig(outfile)
else:
    plt.show()
