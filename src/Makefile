build:
	nvcc pingpong.o main.cu -g -O2 -arch=sm_90 -lnuma
openblas:
	nvcc main.cu -DOPENBLAS -g -O2 -arch=sm_90 -lnuma -lcublas -I/users/lfusco/code/OpenBLAS/installation/include -L/users/lfusco/code/OpenBLAS/installation/lib -lopenblas
unopt:
	nvcc -g -O0 -arch=sm_90 -lcublas main.cu
ptx:
	nvcc -g -O2 -arch=sm_90 -ptx -lcublas main.cu -o main.ptx
pre:
	nvcc -g -O2 -arch=sm_90 -E -lcublas main.cu
