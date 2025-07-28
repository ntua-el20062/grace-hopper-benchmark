build:
	nvcc pingpong.o main.cu -g -O2 -arch=sm_90 -lnuma -lcublas
unopt:
	nvcc pingpong.o main.cu -g -O0 -arch=sm_90 -lnuma
ptx:
	nvcc -g -O2 -arch=sm_90 -ptx -lcublas main.cu -o main.ptx
pre:
	nvcc -g -O2 -arch=sm_90 -E -lcublas main.cu

