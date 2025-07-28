#pragma once

#include "base_functions.hpp"
#include <iostream>
#include "base_kernels.cuh"
#include "measurement.hpp"
#include "constants.hpp"

#define NUM_THREADS 72

constexpr size_t HOST_ID = (size_t) 0;
constexpr size_t DEVICE_ID = (size_t) -1;
constexpr size_t REMOTE_DEVICE_ID = (size_t) -2;

class SpinLock {
    std::atomic_flag locked;
public:
    SpinLock();
    void lock();
    void unlock();
};

double latency_function(uint8_t *in, size_t n_elem) {
    in = in+(n_elem-1)*(2*CACHELINE_SIZE);
    uint64_t start = get_cpu_clock();
    // for (size_t i = 0; i < n_elem; i+=8) {
    //     asm volatile("ldr %0, [%1];"
    //                  "ldr %0, [%1];"
    //                  "ldr %0, [%1];"
    //                  "ldr %0, [%1];"

    //                  "ldr %0, [%1];"
    //                  "ldr %0, [%1];"
    //                  "ldr %0, [%1];"
    //                  "ldr %0, [%1];" : "=r" (in) : "r" ((uint8_t **) in) :);
    // }
    for (size_t i = 0; i < n_elem; ++i) {
        in = (uint8_t *) *((uint64_t *) in); //take the value of in, that is a mem location of a uint8, pretend like it holds a mem location of a uint64 as value, take that uint 64 value, and pretend like it is a uint8 value, assign it to in
					     //(uint64_t)* in: this treats the byte pointer in as a pointer to a 64-bit unsigned integer. This means it reads 8 bytes starting at the address pointed by in
    } //this for loop is a pointer chase, where each 8-byte block at the current address holds the address of the next element
    uint64_t end = get_cpu_clock();

    return get_elapsed_milliseconds_clock(start, end);
}

double latency_write_function(uint8_t *in, size_t n_elem) {    
    uint64_t start = get_cpu_clock();
    for (size_t i = 0; i < n_elem/8; ++i) {
        asm volatile("ldr %0, [%1];"
                     "str x0, [%1, #8];"
                     "ldr %0, [%1];"
                     "str x0, [%1, #8];"
                     "ldr %0, [%1];"
                     "str x0, [%1, #8];"
                     "ldr %0, [%1];"
                     "str x0, [%1, #8];"

                     "ldr %0, [%1];"
                     "str x0, [%1, #8];"
                     "ldr %0, [%1];"
                     "str x0, [%1, #8];"
                     "ldr %0, [%1];"
                     "str x0, [%1, #8];"
                     "ldr %0, [%1];" 
                     "str x0, [%1, #8];" : "=r" (in) : "r" ((uint8_t **) in) :);
    }
    //"str x0, [%1, #8];" : "=r" (in) : "r" ((uint8_t **) in) :):  stores the contents of register x0 into memory at the address given by the pointer in %1, plus an 8-byte offset
    //in cast to a pointer to pointer to uint8_t, passed in a register
    uint64_t end = get_cpu_clock();

    return get_elapsed_milliseconds_clock(start, end);
}

enum ThreadCommand {
    WRITE,
    READ,
    INVALIDATE,
    NONE,
    READ_TEST,
    WRITE_TEST,
    COPY_TEST,
    MEMSET_TEST,
    MEMCPY_TEST,
    CLOCK_TEST,
    LATENCY_TEST,
    LATENCY_WRITE_TEST,
    INFINITE_READ,
    TERMINATE
};

struct thread_data {
    std::unique_ptr<std::thread> t{nullptr};
    size_t t_id;
    SpinLock tx_mutex, rx_mutex;
    ThreadCommand command;
    uint8_t *buffer;
    uint8_t *second_buffer;
    size_t size;
    uint64_t start_time;
    uint64_t end_time;
    size_t n_iter;
} __attribute__((aligned(128)));

thread_data thread_array[NUM_THREADS]; //array of worker threads
uint8_t cache_filler[(int) ((double)CPU_L3_CACHE * 2.1f)];

__attribute__((noinline)) __host__ __device__ void dumb_copy(volatile uint8_t *source, volatile uint8_t *target, size_t size) { //can be run both on host ad device, volatile:no compiler optimizations
    for (size_t i = 0; i < size; i += 64) {
        target[i] = source[i];
    }
}

void prepare_memory(ThreadCommand command, uint8_t *buffer, size_t size) {
    switch (command) {
        case WRITE: {
            //dumb_copy(buffer, buffer, size);
            for (size_t i = 0; i < size; i += 64) {
                asm volatile("ldr x0, [%0];"
                             "str x0, [%0]":: "r" (&buffer[i]) : "x0" );
            } //load and store into the same address, read and write at cache line granularity
            break;
        }
        case READ: {
            for (size_t i = 0; i < size; i += 64) {
                asm volatile("ldr x0, [%0]" :: "r" (&buffer[i]) : "x0" ); //load from memory location at &buffer[i]
            }
            break;
        }
        case INVALIDATE: {
            for (size_t i = 0; i < sizeof(cache_filler); i += 64) {
                asm volatile("ldr x0, [%0]" :: "r" (&cache_filler[i]) : "x0", "memory"); //evict data from cache by loading the bigger cache filler
            }
            break;
        }
    }
    //asm volatile("dmb sy" ::: "memory");
}

__global__ void device_write_preparation_kernel(uint8_t *a, size_t size) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (size_t i = tid * 32; i < size; i += blockDim.x * gridDim.x * 32) {
        a[i] = 0;
    }
}

__global__ void device_read_preparation_kernel(uint8_t *a, size_t size) {
    uint8_t dummy[2];
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (size_t i = tid * 32; i < size; i += blockDim.x * gridDim.x * 32) {
        dummy[i%2] = a[i];
    }

    if (tid == (size_t)-1) { //this will never be true because tid is unsigned and non negative, so we use it so again the compiler doesnt optimize the reads and keep dummy[0] used, if i hadnt that the compiler would know that the dummy is not used therefore eliminated sompletely the writes during compilation
        printf("%u", dummy[0]);
    }
}

void dispatch_command(size_t t_id, ThreadCommand command, uint8_t *buffer, size_t size, uint8_t *second_buffer=nullptr, size_t n_iter=0) {
    if (t_id == DEVICE_ID || t_id == REMOTE_DEVICE_ID) {
        if (t_id == REMOTE_DEVICE_ID) {
            cudaSetDevice(1);
        }
        switch (command) {
            case WRITE: {
                device_write_preparation_kernel<<<264, 1024>>>(buffer, size);
                cudaDeviceSynchronize();
                break;
            }
            case READ: {
                device_read_preparation_kernel<<<264, 1024>>>(buffer, size);
                cudaDeviceSynchronize();
                break;
            }
        }
        if (t_id == REMOTE_DEVICE_ID) {
            cudaSetDevice(0);
        }
    } else { //if this was called from a main cpu thread, the worker threads wait for work
        thread_data *cur = &thread_array[t_id];
        cur->command = command;
        cur->buffer = buffer;
        cur->second_buffer = second_buffer;
        cur->n_iter = n_iter;
        cur->size = size;
        cur->rx_mutex.lock(); //main thread, when the worker thread finishes its job and calls rx_mutex.unlock(), the main thread will be able to proceed (by unlocking the mutex it's currently waiting) 
        cur->tx_mutex.unlock(); // send command from main to workers->tx from main to worker start working now, locked by thread when it starts working, and when started unlocked by main
        cur->rx_mutex.lock(); // main waits until worker thread is done -> rx from workers to main i finished, locked by main, unlocked by workers:i finished working
        cur->rx_mutex.unlock(); //pair the first lock
    }
}

uint64_t run_test(ThreadCommand test, size_t n_threads, size_t initial_thread, uint8_t *buffer, uint8_t *second_buffer, size_t n_elems, uint64_t *end_times, bool skip_waiting=false) {
    //assert(n_threads + initial_thread <= NUM_THREADS);
    size_t n_cachelines = n_elems / 8; // 8 doubles per cacheline
    size_t per_thread_n_cachelines = n_cachelines / n_threads;
    // size_t remainder = n_cachelines % n_threads;
    uint64_t nominal_start_time = get_cpu_clock() + 1000000; // cur + 1ms
    for (size_t i = 0; i < n_threads; ++i) {
        thread_data *cur = &thread_array[i+initial_thread];
        cur->command = test;
        cur->buffer = buffer;
        cur->second_buffer = second_buffer;
        // cur->size = (per_thread_n_cachelines + (i < remainder ? 1 : 0)) * 8;
        cur->size = per_thread_n_cachelines * 8;
        cur->start_time = nominal_start_time;
        buffer += cur->size * sizeof(double); //buffer moved forward for next thread
        second_buffer += cur->size * sizeof(double); //buffer moved forward for next thread
    }


    for (size_t i = 0; i < n_threads; ++i) { //only from main
        thread_data *cur = &thread_array[i+initial_thread];
        cur->rx_mutex.lock(); //main thread locks the receive mutex, so it can later block waiting for the thread to signal completion
        cur->tx_mutex.unlock(); // send command from main to worker start working now
    }


    if (!skip_waiting) {
	std::cout << "num threads:" << n_threads << std::endl;
        for (size_t i = 0; i < n_threads; ++i) {
            thread_data *cur = &thread_array[i+initial_thread];
            cur->rx_mutex.lock(); // wait until thread is done
            cur->rx_mutex.unlock();
            end_times[i] = cur->end_time;

        }

    }
    std::cout << "nominal start time:" << nominal_start_time << std::endl;

    return nominal_start_time;
}

double time_test(ThreadCommand test, size_t n_threads, size_t initial_thread, uint8_t *buffer, uint8_t *second_buffer, size_t n_elems) { 
    uint64_t end_times[n_threads];
    uint64_t nominal_start_time = run_test(test, n_threads, initial_thread, buffer, second_buffer, n_elems, end_times);

    uint64_t max_end = 0;
    for (size_t i = 0; i < n_threads; ++i) {
        max_end = std::max(max_end, end_times[i]);
        // max_end += end_times[i];
    }
    // max_end /= n_threads;

    return get_elapsed_milliseconds_clock(nominal_start_time, max_end);
}

void run_clock_offset_test(size_t n_threads, size_t initial_thread) {
    uint64_t end_times[n_threads];
    std::cout << "CLOCK OFFSET BEFORE" << std::endl;
    run_test(CLOCK_TEST, n_threads, initial_thread, nullptr, nullptr, 0, end_times);
    std::cout << "CLOCK OFFSET AFTER" << std::endl;

    for (size_t i = 0; i < n_threads; ++i) {
        std::cout << end_times[i] << std::endl;
    }
        std::cout << "CLOCK OFFSET END" << std::endl;

}

void invalidate_all(uint8_t *buffer, size_t size) {
    prepare_memory(WRITE, buffer, size);        // put in OWNED state
    prepare_memory(READ, cache_filler, sizeof(cache_filler));   // invalidate locally
}

uint64_t thread_write_function(uint64_t start_time, double *a, size_t n_elems) {
    a -= 2;
    while (get_cpu_clock() < start_time) {}
    for (size_t i = 0; i < n_elems; i += 8) {
        asm volatile("stp x0, x1, [%0, #16];"
                    "stp x0, x1, [%0, #32];"

                    "stp x0, x1, [%0, #48];"
                    "stp x0, x1, [%0, #64]!;" :: "r" (a) : "x0", "x1");
    } //store pair x0, x1 to memory locations offsets a+16, a+32, a+64
    return get_cpu_clock();
}

uint64_t thread_memset_function(uint64_t start_time, double *a, size_t n_elems) {
    while (get_cpu_clock() < start_time) {}
    memset(a, 0xff, n_elems * sizeof(double)); // 0xff since setting to 0x00 does weird stuff
    return get_cpu_clock();
}

uint64_t thread_read_function(uint64_t start_time, double *a, size_t n_elems) {
    a -= 2;
    while (get_cpu_clock() < start_time) {}
    for (size_t i = 0; i < n_elems; i += 8) {
        asm volatile("ldp x0, x1, [%0, #16];"
                    "ldp x0, x1, [%0, #32];"

                    "ldp x0, x1, [%0, #48];"
                    "ldp x0, x1, [%0, #64]!;" :: "r" (a) : "x0", "x1");
    }
    return get_cpu_clock();
}

void infinite_thread_read_function(double *a, size_t n_elems) {
    while (true) {
        for (size_t i = 0; i < n_elems; i += 8) {
            asm volatile("ldp x0, x1, [%0, 0];"
                        "ldp x0, x1, [%0, #16];"

                        "ldp x0, x1, [%0, #32];"
                        "ldp x0, x1, [%0, #48];" :: "r" (&a[i]) : "x0", "x1");
        }
    }
}

uint64_t thread_copy_function(uint64_t start_time, double *a, double *b, size_t n_elems) {
    a -= 2;
    b -= 2;
    while (get_cpu_clock() < start_time) {}
    for (size_t i = 0; i < n_elems; i += 8) {
        // b[i] = a[i];
        // b[i + 1] = a[i + 1];
        // b[i + 2] = a[i + 2];
        // b[i + 3] = a[i + 3];

        // b[i + 4] = a[i + 4];
        // b[i + 5] = a[i + 5];
        // b[i + 6] = a[i + 6];
        // b[i + 7] = a[i + 7];

        asm volatile("ldp x0, x1, [%0, #16];" //load pair x0,x1 from a+16
                    "stp x0, x1, [%1, #16];"  //store pair at b+16
                    "ldp x2, x3, [%0, #32];"
                    "stp x2, x3, [%1, #32];"

                    "ldp x4, x5, [%0, #48];"
                    "stp x4, x5, [%1, #48];"
                    "ldp x6, x7, [%0, #64]!;"
                    "stp x6, x7, [%1, #64]!;" :: "r" (a), "r" (b) : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7");
    }
    return get_cpu_clock();
}

uint64_t thread_memcpy_function(uint64_t start_time, double *a, double *b, size_t n_elems) {
    while (get_cpu_clock() < start_time) {}
    memcpy(b, a, n_elems * sizeof(double));
    return get_cpu_clock();
}

void thread_function(thread_data *t_info) { //what worker threads will run!!!!!!
    for (;;) {
        t_info->tx_mutex.lock(); // wait until main sends command, when main does unlock the tx_mutex, the worker thread get notified and immediately resumes execution after the lock() call
        switch (t_info->command) {
            case TERMINATE:
                return;
            case READ:
            case WRITE:
            case INVALIDATE:
                prepare_memory(t_info->command, t_info->buffer, t_info->size);
                break;
            case READ_TEST:
                t_info->end_time = thread_read_function(t_info->start_time, (double *)t_info->buffer, t_info->size);
                break;
            case WRITE_TEST:
                t_info->end_time = thread_write_function(t_info->start_time, (double *)t_info->buffer, t_info->size);
                break;
            case COPY_TEST:
                t_info->end_time = thread_copy_function(t_info->start_time, (double *)t_info->buffer, (double *) t_info->second_buffer, t_info->size);
                break;
            case MEMSET_TEST:
                t_info->end_time = thread_memset_function(t_info->start_time, (double *)t_info->buffer, t_info->size);
                break;
            case MEMCPY_TEST:
                t_info->end_time = thread_memcpy_function(t_info->start_time, (double *)t_info->buffer, (double *)t_info->second_buffer, t_info->size);
                break;
            case CLOCK_TEST: {
                uint64_t cur;
                do {
                    cur = get_cpu_clock();
                } while (cur < t_info->start_time);
                t_info->end_time = cur;
                break;
            }
            case LATENCY_TEST: {
                double *times = (double *) t_info->second_buffer;
                size_t n_iter = t_info->n_iter;
                // memcpy(cache_filler, cache_filler, sizeof(cache_filler));
                for (size_t i = 0; i < n_iter; ++i) {
                    times[i] = latency_function(t_info->buffer, t_info->size);
                    // for (size_t i = 0; i < sizeof(cache_filler); i += 64) {
                    //     asm volatile("ldr x0, [%0]" :: "r" (&cache_filler[i]) : "x0", "memory");
                    // }
                }
                break;
            }
            case LATENCY_WRITE_TEST: {
                double *times = (double *) t_info->second_buffer;
                size_t n_iter = t_info->n_iter;
                for (size_t i = 0; i < n_iter; ++i) {
                    times[i] = latency_write_function(t_info->buffer, t_info->size);
                    // for (size_t i = 0; i < sizeof(cache_filler); i += 64) {
                    //     asm volatile("ldr x0, [%0]" :: "r" (&cache_filler[i]) : "x0", "memory");
                    // }
                }
                break;
            }
            case INFINITE_READ:
                infinite_thread_read_function((double *) t_info->buffer, t_info->size);
                break;
        }
        t_info->rx_mutex.unlock(); // signal completion
    }
}

void init_thread_array(int main_thread) { 
    //assert(main_thread >= NUM_THREADS);
    for (size_t i = 0; i < NUM_THREADS; ++i) { //for loop for main thread
        thread_data *cur = &thread_array[i];
        cur->t_id = i;
	std::cout << "INITIALLIZED cur tid" << cur->t_id << std::endl;
        cur->start_time = 0;
        cur->tx_mutex.lock(); //lock tx mutex so worker thread will block on it at startup and wait for tx unlock = main-> worker "start working now"

	// Spawn the worker thread running 'thread_function' with the thread_data pointer.
        cur->t = std::make_unique<std::thread>(thread_function, cur);

	//set cpu affinity to avoid migrations
	cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(i, &cpuset); 
	/*
	cur->tx_mutex.lock():
	You lock the mutex before starting the worker thread. This is critical because your worker thread starts by trying to lock() this same mutex, which causes it to block immediately. This prevents the worker from running commands before the main thread tells it to.

	Thread creation:
	Each worker thread runs thread_function with a pointer to its thread_data.

	CPU affinity:
	Each thread is pinned to a distinct CPU core to minimize context switching and maximize cache locality.
	*/

        pthread_setaffinity_np(cur->t->native_handle(), sizeof(cpu_set_t), &cpuset);
        struct sched_param param;
        param.sched_priority = 1;
        pthread_setschedparam(cur->t->native_handle(), SCHED_FIFO, &param);
    }
}


void terminate_threads() {
    printf("terminating...\n");
    fflush(stdout);
    for (size_t i = 0; i < NUM_THREADS; ++i) { //for each worker thread
        thread_data *cur = &thread_array[i];
	std::cout << "thread:" << cur->t_id << std::endl;
        cur->command = TERMINATE; //the main cur thread sets command to ternimate
	    
        cur->tx_mutex.unlock(); //tells worket thread to start job
            
	//cur->t->join(); //waits for worker thread to finish
        if (cur->t && cur->t->joinable()) {
		cur->t->join();
		std::cout << "joined" << std::endl;

	} else {
    		std::cerr << "Thread not joinable or null for thread " << i << std::endl;
	}

    }
    std::cout << "DONE" << std::endl;
}


/*d (the thread running your program’s control logic) dispatches commands to worker threads.
Each worker thread corresponds to one thread_data entry in thread_array and runs independently.
The main thread controls each worker thread by:
Setting command and parameters in that worker’s thread_data.
Unlocking that worker’s tx_mutex to signal it to start processing.
Waiting on the worker’s rx_mutex to know when the worker finished.
So the relationship is one main thread controlling many worker threads — each worker thread listens on its own pair of mutexes (tx_mutex and rx_mutex), processing commands sequentially as the main thread unlocks its tx_mutex.
*/
