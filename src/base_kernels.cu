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

    asm volatile (
        "ld.global.f32 %0, [%1];"
        : "=f"(local) : "l"(target_address));
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