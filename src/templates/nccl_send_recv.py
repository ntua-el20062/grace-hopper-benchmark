base = 'send_recv<NCCL_ALLOC<{}>, NCCL_ALLOC<{}>>(rank, world_size, n_iter, n_bytes, comm, stream, "{}_{}/" + std::to_string(n_bytes));'

memories = [
    ('ddr', 'CUR_CPU'),
    ('hbm', 'CUR_GPU'),
    ('ddr_remote', 'OTHER_CPU'),
    ('hbm_remote', 'OTHER_GPU')
]

import itertools
import os

for a, b in itertools.product(memories, memories):
    print(base.format(a[1], b[1], a[0], b[0]))