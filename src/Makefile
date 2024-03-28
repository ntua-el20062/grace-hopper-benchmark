build:
	nvcc pingpong.o main.cu -g -O2 -arch=sm_90 -lnuma -lcublas
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
	nvcc -arch=sm_90 -O2 -o nccl.out pingpong.o nccl.cpp -I/user-environment/env/default/include -L/user-environment/env/default/lib64 -L/user-environment/env/default/lib -lcudart -lmpi -lnccl -lnuma -lmpi -L/user-environment/linux-sles15-neoverse_v2/gcc-12.3.0/cray-gtl-8.1.28-zrpwudcmtmevtnjgtbug7ykbdyxkqmrn/lib -lmpi_gtl_cuda