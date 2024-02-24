base = 'cublas_gemm_template<{}, {}, {}>(n_iter, n_bytes, {}, {}, {}, "{}_{}_{}/" + std::to_string(n_bytes));'

memories = [
    ('ddr', 'MmapDataFactory', 'HOST_ID'),
    ('hbm', 'MmapDataFactory', 'DEVICE_ID'),
    ('ddr_remote', 'NumaDataFactory<1>', 'HOST_ID'),
    ('hbm_remote', 'MmapDataFactory', 'REMOTE_DEVICE_ID')
]

import itertools
import os

for a, b, c in itertools.product(memories, memories, memories):
    try:
        os.mkdir(f'../results/apps/gemm/cublas/{a[0]}_{b[0]}_{c[0]}')
    except:
        pass
    print(base.format(a[1], b[1], c[1], a[2], b[2], c[2], a[0], b[0], c[0]))