#!/usr/bin/env python
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) == 2:
    print(sys.argv)
    result_fname = sys.argv[1]
else:
    result_fname = 'results.txt'

# Read results

results = {}
with open(result_fname, 'r') as flopfile:
    flopreader = csv.reader(flopfile)
    for row in flopreader:
        n = int(row[0])
        try:
            results[n] = [float(r) for r in row[1:]]
        except:
            print(row)
            raise

run_names = [
    "y[:] = x[:]",
    "y[:] = a x[:]",
    "y[:] = x[:] + x[:]",
    "y[:] = x[:] + y[:]",
    "y[:] = a x[:] + y[:]",
    "y[:] = a x[:] + b y[:]",
    "y[1:] = x[1:] + x[:-1]",
    "y[8:] = x[8:] + x[:-8]",
    "GPU: y[:] = a * x[:] + y[:]",
]

x = np.array([n for n in results])

y_results = []
for i in range(len(results[x[0]])):
    y_results.append(np.array([results[n][i] for n in results]))

# first eight diff8 results are garbage, so set to zero
y_results[15][:8] = 0.

# Plot results

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.set_title('Performance (GFLOP/s) vs vector length')
ax2.set_title('Bandwidth (GB/s) vs vector length')

ax1.set_ylabel('GFLOP / sec')
ax2.set_ylabel('GB / sec')

for ax in (ax1, ax2):
    ax.set_xscale('log')
    ax.set_xlabel('Vector length')

# Skip GPU results (y_results[8][:])
for y, name in zip(y_results[:6], run_names[1:7]):
    ax1.plot(x, y / 1e9, label=name)

# Skip first 8 nonsense diff8 timings
ax1.plot(x[8:], y_results[6][8:] / 1e9, label=run_names[7])

# No, don't skip GPU!
ax1.plot(x[:], y_results[7][:] / 1e9, label=run_names[8])

ax1.legend()


for y, name in zip(y_results[9:15], run_names[1:7]):
    ax2.plot(x, y / 1e9, '.', label=name)

# diff8 again
ax2.plot(x[8:], y_results[15][8:] / 1e9, '.')

ax2.legend()

plt.show()
