build:
	nvcc main.cu -g -O2 -arch=sm_90 -lcublas
# nvcc main.cu -g -O2 -arch=sm_90 -I/users/lfusco/libs/arm_compute-v24.01-bin-linux-arm64-v8.2-a-neon-cl -I/users/lfusco/libs/arm_compute-v24.01-bin-linux-arm64-v8.2-a-neon-cl/include -L/users/lfusco/libs/arm_compute-v24.01-bin-linux-arm64-v8.2-a-neon-cl/lib/arm64-v8.2-a-neon-cl -larm_compute -lcublas
unopt:
	nvcc -g -O0 -arch=sm_90 -lcublas main.cu
ptx:
	nvcc -g -O2 -arch=sm_90 -ptx -lcublas main.cu -o main.ptx
pre:
	nvcc -g -O2 -arch=sm_90 -E -lcublas main.cu
