#!/usr/bin/env python
import csv

import matplotlib.pyplot as plt
import numpy as np

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

fig, (ax1, ax2) = plt.subplots(2, 1)

for ax in (ax1, ax2):
    ax.set_xscale('log')

for y in y_results[:4]:
    ax1.plot(x, y)

for y in y_results[4:]:
    ax2.plot(x, y)

plt.show()
