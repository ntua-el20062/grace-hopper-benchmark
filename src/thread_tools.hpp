#pragma once

#define NUM_THREADS 3

class SpinLock {
    std::atomic_flag locked = ATOMIC_FLAG_INIT;
public:
    void lock() {
        while (locked.test_and_set(std::memory_order_acquire)) {}
    }
    void unlock() {
        locked.clear(std::memory_order_release);
    }
};

struct sized_buffer {
    void *buffer;
    size_t size;
};

enum ThreadCommand {
    WRITE,
    READ,
    INVALIDATE,
    TERMINATE
};


struct thread_data {
    std::unique_ptr<std::thread> t{nullptr};
    SpinLock tx_mutex, rx_mutex;
    ThreadCommand command;
    uint8_t *buffer;
    size_t size;
};

thread_data thread_array[NUM_THREADS];

void prepare_memory(ThreadCommand command, uint8_t *buffer, size_t size) {
    switch (command) {
        case WRITE: {
            for (size_t i = 0; i < size; ++i) {
                buffer[i] = (uint8_t) i;
            }
            break;
        }
        case READ: {
            size_t acc = 0;
            for (size_t i = 0; i < size; ++i) {
                acc += buffer[i];
            }
            size = acc; // dummy usage of acc
            break;
        }
        case INVALIDATE: {
            for (size_t i = 0; i < size; i += 64) {
                asm volatile("dc civac, %0" : : "r"(&buffer[i]) : "memory");
            }
            break;
        }
    }
}

void dispatch_memory_preparation(size_t t_id, ThreadCommand command, uint8_t *buffer, size_t size) {
    if (t_id == 0) {
        prepare_memory(command, buffer, size);
    } else {
        thread_data *cur = &thread_array[t_id - 1];
        cur->command = command;
        cur->buffer = buffer;
        cur->size = size;
        cur->rx_mutex.lock();
        cur->tx_mutex.unlock(); // send command
        cur->rx_mutex.lock(); // wait until thread is done
        cur->rx_mutex.unlock();
    }
}

void thread_function(thread_data *t_info) {
    for (;;) {
        t_info->tx_mutex.lock(); // wait until main sends command
        switch (t_info->command) {
            case TERMINATE:
                return;
            default:
                prepare_memory(t_info->command, t_info->buffer, t_info->size);
        }
        t_info->rx_mutex.unlock(); // signal completion
    }
}

void init_thread_array() {
    for (size_t i = 0; i < NUM_THREADS; ++i) {
        thread_data *cur = &thread_array[i];
        cur->tx_mutex.lock();
        cur->t = std::make_unique<std::thread>(thread_function, cur);
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(i + 1, &cpuset); 

        pthread_setaffinity_np(cur->t->native_handle(), sizeof(cpu_set_t), &cpuset);
    }
}

void terminate_threads() {
    for (size_t i = 0; i < NUM_THREADS; ++i) {
        thread_data *cur = &thread_array[i];

        cur->command = TERMINATE;
        cur->tx_mutex.unlock();
        cur->t->join();
    }
}