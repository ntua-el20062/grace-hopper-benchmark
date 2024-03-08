import os

osu_base = '/users/lfusco/bin/libexec/osu-micro-benchmarks/mpi'
cuda = ' -d cuda D D'
resdir = 'results/osu'

one_sided_benchmarks = ['osu_get_bw', 'osu_get_latency'] osu_mbw_mr
collective_benchmarks = ['osu_allgather', 'osu_allreduce', 'osu_alltoall']

def run_one_sided():
    for b in one_sided_benchmarks:
        os.system(f'srun -N 2 {osu_base}/one-sided/{b} | tee {resdir}/{b}')
        os.system(f'srun -N 2 {osu_base}/one-sided/{b} {cuda} | tee {resdir}/{b}_cuda')

def run_collective():
    i = 2
    while i <= 128:
        for b in collective_benchmarks:
            os.system(f'srun -N {i} {osu_base}/collective/{b} | tee {resdir}/{b}_{i}')
            os.system(f'srun -N {i} {osu_base}/collective/{b} {cuda} | tee {resdir}/{b}_{i}_cuda')
        i = i * 2

# run_one_sided()
run_collective()