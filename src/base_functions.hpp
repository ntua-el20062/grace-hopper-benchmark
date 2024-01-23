#pragma once

#include <cuda/atomic>

alignas(64) volatile uint8_t cacheline_array[64];

__attribute__((always_inline)) inline void cacheline_write_function(uint8_t *out, size_t len) {
    uint8_t *itr = out;
    for (size_t i = 0; i < len/(64*16); ++i) {
        // write after 8 bytes to keep the "next address" intact
        // then, load the "next address"
        itr[8] = 0; itr = *((uint8_t **) itr); 
        itr[8] = 0; itr = *((uint8_t **) itr);
        itr[8] = 0; itr = *((uint8_t **) itr);
        itr[8] = 0; itr = *((uint8_t **) itr);
        itr[8] = 0; itr = *((uint8_t **) itr);
        itr[8] = 0; itr = *((uint8_t **) itr);
        itr[8] = 0; itr = *((uint8_t **) itr);
        itr[8] = 0; itr = *((uint8_t **) itr);

        itr[8] = 0; itr = *((uint8_t **) itr);
        itr[8] = 0; itr = *((uint8_t **) itr);
        itr[8] = 0; itr = *((uint8_t **) itr);
        itr[8] = 0; itr = *((uint8_t **) itr);
        itr[8] = 0; itr = *((uint8_t **) itr);
        itr[8] = 0; itr = *((uint8_t **) itr);
        itr[8] = 0; itr = *((uint8_t **) itr);
        itr[8] = 0; itr = *((uint8_t **) itr);
    }
    //asm volatile("dmb st" ::: "memory");
}

__attribute__((always_inline)) inline void cacheline_read_function(uint8_t *in, size_t len) {
    for (size_t i = 0; i < len/(64*16); ++i) {
        asm volatile("ldr %0, [%1];"
                     "ldr %0, [%1];"
                     "ldr %0, [%1];"
                     "ldr %0, [%1];"
                     "ldr %0, [%1];"
                     "ldr %0, [%1];"
                     "ldr %0, [%1];"
                     "ldr %0, [%1];"

                     "ldr %0, [%1];"
                     "ldr %0, [%1];"
                     "ldr %0, [%1];"
                     "ldr %0, [%1];"
                     "ldr %0, [%1];"
                     "ldr %0, [%1];"
                     "ldr %0, [%1];"
                     "ldr %0, [%1];" : "=r" (in) : "r" ((uint8_t **) in) :);
    }
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