build:
	nvcc pingpong.o main.cu -g -O2 -arch=sm_90 -lnuma
openblas:
	nvcc main.cu -DOPENBLAS -g -O2 -arch=sm_90 -lnuma -lcublas -I/users/lfusco/code/OpenBLAS/installation/include -L/users/lfusco/code/OpenBLAS/installation/lib -lopenblas
unopt:
	nvcc pingpong.o main.cu -g -O0 -arch=sm_90 -lnuma
ptx:
	nvcc -g -O2 -arch=sm_90 -ptx -lcublas main.cu -o main.ptx
pre:
	nvcc -g -O2 -arch=sm_90 -E -lcublas main.cu
nccl:
	# mpicxx -O2 -o nccl.out pingpong.o nccl.cpp -I/user-environment/env/default/include -L/user-environment/env/default/lib64 -L/user-environment/env/default/lib -lcudart -lmpi -lnccl -lnuma -lmpi
	nvcc -arch=sm_90 -O2 -o nccl.out pingpong.o nccl.cpp -I/user-environment/env/default/include -L/user-environment/env/default/lib64 -L/user-environment/env/default/lib -lcudart -lmpi -lnccl -lnuma -lmpi