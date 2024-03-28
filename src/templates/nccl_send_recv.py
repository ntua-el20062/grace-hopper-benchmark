base = 'send_recv<{}, {}>(rank, world_size, n_iter, n_bytes, comm, stream, "{}_{}/" + std::to_string(n_bytes));'

memories = [
    ('ddr', 'NUMA_ALLOC<CUR_CPU>'),
    ('hbm', 'NUMA_ALLOC<CUR_GPU>'),
    ('ddr_remote', 'NUMA_ALLOC<OTHER_CPU>'),
    ('hbm_remote', 'NUMA_ALLOC<OTHER_GPU>'),
    ('cuda', 'CUDA_ALLOC<CUR_GPU>'),
]

import itertools
import os

for d in ['mono', 'bi']:
    for m in ['mpi', 'nccl']:
        for i in [1,2,4]:
            for a, b in itertools.product(memories, memories):
                # print(base.format(a[1], b[1], a[0], b[0]))
                os.system(f'mkdir "/users/lfusco/code/gh_benchmark/src/results/nccl/group_send_recv/{d}/{m}/{i}/{a[0]}_{b[0]}"')