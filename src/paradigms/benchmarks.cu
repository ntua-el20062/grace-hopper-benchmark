#include "kernels.cu"
#include "../measurement.hpp"

float kernel_invocation_benchmark(int N, int M) {
    float time = time_function_execution([N, M](){
        dynamic_work_kernel<<<N/32, 32>>>();
        cudaDeviceSynchronize();
        dynamic_work_kernel<<<M/32, 32>>>();
        cudaDeviceSynchronize();
    });

    return time;
}

float flattened_parallelism_benchmark(int N, int M) {
    cub::GridBarrierLifetime gbar;
    gbar.Setup((N + M) * 100);
    void *args[] = {(void *) &N, (void *) &M, (void *) &gbar};
    float time = time_kernel_execution((void *) flattened_parallelism_kernel, (N + M) / 32, 32, args, 0, 0);

    return time;
}

float dynamic_parallelism_benchmark(int N, int M) {
    void *args[] = {(void *) &N, (void *) &M};
    float time = time_kernel_execution((void *) dynamic_parallelism_kernel, 1, 1, args, 0, 0);

    return time;
}

float cooperative_parallelism_benchmark(int N, int M) {
    void *args[] = {(void *) &N, (void *) &M};
    float time = time_cooperative_kernel_execution((void *) cooperative_parallelism_kernel, args, 0, 0);

    return time;
}