#!/usr/bin/env python
import csv

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

x = np.array([n for n in results])

y_results = []
for i in range(len(results[1])):
    y_results.append(np.array([results[n][i] for n in results]))

# Plot results

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.set_title('Performance (GFLOP/s) vs vector length')
ax2.set_title('Bandwidth (GB/s) vs vector length')

ax1.set_ylabel('GFLOP / sec')
ax2.set_ylabel('GB / sec')

for ax in (ax1, ax2):
    ax.set_xscale('log')
    ax.set_xlabel('Vector length')

for y in y_results[:4]:
    ax1.plot(x, y / 1e9)

for y in y_results[4:]:
    ax2.plot(x, y / 1e9)

plt.show()
