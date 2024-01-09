#include <cuda/atomic>

template <typename T, unsigned int STRIDE>
void strided_write_function(T *out, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        out[i * STRIDE] = 0;
    }
    asm volatile("" ::: "memory");
}

template <typename T, unsigned int STRIDE>
void strided_copy_function(T *out, const T *in, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        out[i * STRIDE] = in[i * STRIDE];
    }
}

void pointer_chase_function(void *ptr) {
    while (ptr) {
        ptr = *((void **) ptr);
    }
}

void atomic_cas_pointer_chase_function(void *ptr) {
    while (ptr) {
        ptr = (void *) __sync_val_compare_and_swap((uint64_t *) ptr, 0, 0);
    }
}

void ping_pong_send_first_function(void *to_send, void *received, void *send_buffer, bool *send_canary, void *recv_buffer, bool *recv_canary, size_t buffer_size) {
    memcpy(send_buffer, to_send, buffer_size);
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system);
    *send_canary = true;
    while (!*recv_canary) {}
    memcpy(received, recv_buffer, buffer_size);
}