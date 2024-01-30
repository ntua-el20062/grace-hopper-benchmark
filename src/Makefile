build:
	nvcc -g -O2 -arch=sm_90 main.cu
unopt:
	nvcc -g -O0 -arch=sm_90 main.cu
# omp:
# 	nvcc -g -O2 -arch=sm_90 -Xcompiler -fopenmp main.cu
ptx:
	nvcc -g -O2 -arch=sm_90 -ptx main.cu -o main.ptx
pre:
	nvcc -g -O2 -arch=sm_90 -E main.cu
