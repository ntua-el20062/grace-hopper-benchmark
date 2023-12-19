#include <cooperative_groups.h>
#include <cuda.h>

namespace cg = cooperative_groups;

/**
 * these functions are meant to evaluate typical memory access pattern from device
 * memory should be prepared in advance in such a way that makes these kernels meaningful
 */

template <typename T, unsigned int STRIDE>
__global__ void strided_write_kernel(T *out) {
    auto tid = cg::this_grid().thread_rank();

    out[tid * STRIDE] = 0;
}

template <typename T, unsigned int STRIDE>
__global__ void strided_read_kernel(const T *in) {
    auto tid = cg::this_grid().thread_rank();
    T local;
    const T *target_address = in + tid;

    assert(!*target_address);
}

template <typename T, unsigned int STRIDE>
__global__ void strided_copy_kernel(T *out, const T *in) {
    auto tid = cg::this_grid().thread_rank();

    out[tid * STRIDE] = in[tid * STRIDE];
}

template <typename T, unsigned int STRIDE>
__global__ void strided_sum_kernel(T *out, const T *a, const T *b) {
    auto tid = cg::this_grid().thread_rank();

    out[tid * STRIDE] = a[tid * STRIDE] + b[tid * STRIDE];
}

__global__ void pointer_chase_kernel(unsigned long long int *ptr) {
    while (ptr) {
        ptr = (unsigned long long int *) *ptr;
    }
}

__global__ void atomic_cas_pointer_chase_kernel(unsigned long long int *ptr) {
    while (ptr) {
        ptr = (unsigned long long int *) atomicCAS(ptr, 0, 0);
    }
}

// LAUNCH WITH ONLY ONE BLOCK!
__global__ void ping_pong_receive_first_kernel(void *to_send, void *received, void *send_buffer, bool *send_canary, void *recv_buffer, bool *recv_canary, size_t buffer_size) {
    auto tid = cg::this_thread_block().thread_rank();

    if (tid == 0) {
        while (!*recv_canary) {}
    }

    __syncthreads();

    for (size_t i = tid; i < buffer_size / sizeof(uint64_t); i += cg::this_thread_block().size()) {
        ((uint64_t *) received)[i] = ((uint64_t *) recv_buffer)[i];
    }

    for (size_t i = tid; i < buffer_size / sizeof(uint64_t); i += cg::this_thread_block().size()) {
        ((uint64_t *) send_buffer)[i] = ((uint64_t *) to_send)[i];
    }

    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system);

    __syncthreads();

    if (tid == 0) {
        *send_canary = true;
    }
}