#include <cooperative_groups.h>
#include "cub/cub/grid/grid_barrier.cuh"

namespace cg = cooperative_groups;

/**
    i = map[0:N]:
        work[i]
        j = map[0:M]:
            work[j]
        work[i]
        k = map[0:K]
            work[k]
        work[i]
*/

__device__ float dummy_work() {
    return 0;
}

__global__ void dynamic_work_kernel() {
    dummy_work();
}

/*
// launched on N threads
__global__ void dynamic_parallelism_nested_kernel(int N, int M, int K) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    dummy_work();

    dynamic_work_kernel<<<M/256, 256>>>();
    cudaDeviceSynchronize();

    dummy_work();

    dynamic_work_kernel<<<K/256, 256>>>();
    cudaDeviceSynchronize();

    dummy_work();
}

// launched on N threads
__global__ void simple_parallelism_nested_kernel(int N, int M, int K) { // inner maps are not unrolled -> outer loop needs to provide high parallelism
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = tid; i < N; i += blockDim.x) {
        dummy_work();

        for (int j = 0; j < M; ++j) {
            dummy_work(, tmp);
        }

        dummy_work();

        for (int k = 0; k < K; ++k) {
            dummy_work(, tmp);
        }

        dummy_work();
    }
}

// launched on N x (M + K) threads
__global__ void flattened_nested_kernel(int N, int M, int K, cub::GridBarrier __gbar) { // strategy here is work repetition
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int i = tid % N;        // which of the outer loop elements I need to compute
    int j = tid / (M + K);  // which of the first level inner loop elements I need to compute
    dummy_work();
    if (j >= 0 && j < M) {
        dummy_work(, tmp);
    }
    __gbar.Sync()

    dummy_work();

    if (j >= M && j < M + K) {
        dummy_work()
    }
}

// launched on fixed number of threads
__global__ void cooperative_flattened_nested_kernel(int N, int M, int K, cub::GridBarrier __gbar) { // strategy here is work repetition
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int i = tid % N;        // which of the outer loop elements I need to compute
    int j = tid / (M + K);  // which of the first level inner loop elements I need to compute
    dummy_work();
    if (j >= 0 && j < M) {
        dummy_work(, tmp);
    }
    __gbar.Sync()

    dummy_work();

    if (j >= M && j < M + K) {
        dummy_work()
    }
}
*/
/////////////////// single depth kernels //////////////////////////

// launched on 1 thread!
__global__ void dynamic_parallelism_kernel(int N, int M) {
    dynamic_work_kernel<<<N/32, 32>>>();
    dynamic_work_kernel<<<M/32, 32>>>();
}

// launched on N + M threads
__global__ void flattened_parallelism_kernel(int N, int M, cub::GridBarrier __gbar) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= 0 && tid < N) {
        dummy_work();
    }

    __gbar.Sync();

    if (tid >= N && tid < N + M) {
        dummy_work();
    }
}

// launched on fixed number of threads
__global__ void cooperative_parallelism_kernel(int N, int M) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto g = cg::this_grid();

    for (int i = tid; i < N; ++i) {
        dummy_work();
    }

    g.sync();

    for (int i = tid; i < M; ++i) {
        dummy_work();
    }
}
