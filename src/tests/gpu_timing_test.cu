#include <stdio.h>

__attribute__((always_inline)) __device__ inline clock_t get_gpu_clock() {
    unsigned long long tsc;

    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tsc));

    return tsc;
}

__global__ void wait_kernel(clock_t *t) {
    clock_t end = get_gpu_clock() + 1000000000;
    while (get_gpu_clock() < end) {}
    *t = get_gpu_clock();
}

int main() {
    cudaEvent_t start, end;
    clock_t t;

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    wait_kernel<<<1, 1>>>(&t);
    cudaEventRecord(end);

    cudaEventSynchronize(end);

    float ms;

    cudaEventElapsedTime(&ms, start, end);

    printf("%f\n", ms);

    return 0;
}