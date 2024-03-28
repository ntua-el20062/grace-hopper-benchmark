base = 'cublas_gemm_template<{}, {}, {}>(n_iter, n_bytes, "{}_{}_{}/" + std::to_string(n_bytes));'

memories = [
    ('ddr', 'HOST_MEM'),
    ('hbm', 'DEVICE_MEM'),
    ('ddr_remote', 'REMOTE_HOST_MEM'),
    ('hbm_remote', 'REMOTE_DEVICE_MEM')
]

types = ['double', 'float', 'half']

import itertools
import os

for a, b, c in itertools.product(memories, memories, [memories[1]]):
    print(base.format(a[1], b[1], c[1], a[0], b[0], c[0]))

for d in types:
    try:
        os.mkdir(f'../results/apps/gemm/cublas/{d}')
    except:
        pass

for a, b, c, d in itertools.product(memories, memories, [memories[1]], types):
    try:
        os.mkdir(f'../results/apps/gemm/cublas/{d}/{a[0]}_{b[0]}_{c[0]}')
    except:
        pass