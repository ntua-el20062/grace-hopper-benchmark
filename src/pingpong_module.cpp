#include <cstdint>
#include <stddef.h>
#include <atomic>

#define FLAG_A 0
#define FLAG_B 1

static __attribute__((always_inline)) inline uint64_t get_cpu_clock() {
    uint64_t tsc;

    asm volatile("isb" : : : "memory");
    asm volatile("mrs %0, cntvct_el0" : "=r"(tsc)); // alternative is cntpct_el0

    return tsc;
}

static __attribute__((always_inline)) inline uint64_t get_cpu_freq() {
    uint64_t freq;

    asm volatile("mrs %0, cntfrq_el0" : "=r"(freq));

    return freq;
}

static double clock_to_milliseconds(uint64_t c) {
    double freq = (double) get_cpu_freq();

    return ((double)(c)) / (freq / 1000.);

}

static double get_elapsed_milliseconds_clock(uint64_t start, uint64_t end) {
    return clock_to_milliseconds(end - start);
}

extern "C" {
    void host_ping_function(std::atomic<uint8_t> *flag, double *time) {
        while (flag->load() == FLAG_B);
        uint8_t expected = FLAG_A;

        uint64_t start = get_cpu_clock();
        for (size_t i = 0; i < 10000; ++i) {
            while (!flag->compare_exchange_strong(expected, FLAG_B, std::memory_order_relaxed, std::memory_order_relaxed)) {
                expected = FLAG_A;
            }
        }
        uint64_t end = get_cpu_clock();
        *time = get_elapsed_milliseconds_clock(start, end);
    }

    void host_pong_function(std::atomic<uint8_t> *flag) {
        uint8_t expected = FLAG_B;
        flag->store(FLAG_A);
        for (size_t i = 0; i < 10000; ++i) {
            while (!flag->compare_exchange_strong(expected, FLAG_A, std::memory_order_relaxed, std::memory_order_relaxed)) {
                expected = FLAG_B;
            }
        }
    }
}