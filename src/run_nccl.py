import os

nccl_dir = '/users/lfusco/code/gh_benchmark/src/nccl.out'

i = 32
while i <= 512:
    os.system(f'srun -N{i} -n{i*4} {nccl_dir}')
    i *= 2