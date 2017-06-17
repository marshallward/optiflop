#!/usr/bin/env python
import csv
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np

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

ai_min, ai_max = -6, 2
ai_grid = 2.**np.arange(ai_min, ai_max + 1)
ai_axis = np.array([Fraction(x).limit_denominator() for x in ai_grid])

ax.set_xscale('log')
ax.set_xlim(2.**ai_min, 2.**ai_max)
ax.set_xticks(ai_axis)
ax.set_xticklabels(ai_axis)

p_min, p_max = -4, 2
P_peak = 8 * 3.3e9
perf_grid = P_peak * 2.**np.arange(p_min, p_max + 1)

ax.set_yscale('log')
ax.set_ylim(P_peak * 2.**(p_min - 0.5), P_peak * 2.**p_max)
ax.set_yticks(perf_grid)
ax.set_yticklabels(['{:.1f}'.format(tk / 1e9) for tk in perf_grid])


for (x, y) in zip(x_ai[:4], y_results[:4]):
    ax.scatter(x, y)

plt.show()
