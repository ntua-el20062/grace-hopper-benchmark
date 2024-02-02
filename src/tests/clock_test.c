#include <stdio.h>

__attribute__((always_inline)) inline unsigned long get_cyclecount() {
    unsigned long value;
    asm volatile("mrs %0, PMCCNTR_EL0" : "=r"(value));  
    return value;
}

int main() {
    printf("%lu\n", get_cyclecount());
}