import os

numa = [0, 1, 4, 12]

for n in numa:
    os.system(f'srun python run_llama33.py {n} 10 > src/results/llama/33/{n}')