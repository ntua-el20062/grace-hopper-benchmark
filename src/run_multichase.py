import os
import math

base = 'NUMA_ALLOC={} srun --cpu-bin=map_cpu:0 /users/lfusco/code/multichase/multichase {} -m {} > {}'

numa_nodes = [0, 1, 4, 12]
memories = ['ddr', 'ddr_remote', 'hbm', 'hbm_remote']

for n, m in zip(numa_nodes, memories):
    i = 4096
    while i <= 2**32:
        print(m, i)
        os.system(base.format(n, '' i, f'/users/lfusco/code/gh_benchmark/src/results/latency/host/scalability/{m}/{i}'))
        os.system(base.format(n, '-c gpu' i, f'/users/lfusco/code/gh_benchmark/src/results/latency/device/scalability/{m}/{i}'))
        i = int(i * math.sqrt(2))