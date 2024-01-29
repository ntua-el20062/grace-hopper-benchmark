build:
	nvcc -g -O2 -arch=sm_90 -rdc=true main.cu
	export OMP_NUM_THREADS=
	export GOMP_CPU_AFFINITY=
omp:
	nvcc -g -O2 -arch=sm_90 -rdc=true -Xcompiler -fopenmp main.cu
	export OMP_NUM_THREADS=64
	export GOMP_CPU_AFFINITY="0-63"
ptx:
	nvcc -g -O2 -arch=sm_90 -rdc=true -ptx main.cu -o main.ptx
pre:
	nvcc -g -O2 -arch=sm_90 -rdc=true -E -Xcompiler -fopenmp main.cu
