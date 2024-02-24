#include <stdio.h>

int main() {
    unsigned long base;

    asm volatile("mrs %0, cntvct_el0" : "=r"(base));
    base += 1000000000;

    #pragma omp parallel
    {
        unsigned long tsc;
        do {
            asm volatile("mrs %0, cntvct_el0" : "=r"(tsc));
        } while (tsc < base);

        printf("%lu\n", tsc);
    }
}