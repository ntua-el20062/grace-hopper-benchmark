template <typename T, unsigned int STRIDE>
void strided_write_function(T *out, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        out[i * STRIDE] = 0;
    }
}

template <typename T, unsigned int STRIDE>
void strided_copy_function(T *out, const T *in, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        out[i * STRIDE] = in[i * STRIDE];
    }
}

void pointer_chase_function(unsigned long long int *ptr) {
    while (ptr) {
        ptr = (unsigned long long int *) *ptr;
    }
}

void atomic_cas_pointer_chase_function(unsigned long long int *ptr) {
    while (ptr) {
        ptr = (unsigned long long int *) __sync_val_compare_and_swap(ptr, 0, 0);
    }
}