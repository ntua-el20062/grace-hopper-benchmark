#include <fcntl.h>
#include <numa.h>
#include <stdio.h>

int main() {
    size_t size = 1UL << 32;
    for (size_t node = 0; node < 36; ++node) {
        printf("%lu\n", node);
        void *data = numa_alloc_onnode(size, node);
        numa_free(data, size);
    }
}