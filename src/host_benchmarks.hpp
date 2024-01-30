#include "base_functions.hpp"
#include "thread_tools.hpp"

template <int TID>
double contiguous_host_modified_host_write_benchmark(size_t n_bytes) {
    size_t size = n_bytes;
    MallocDataFactory data_factory(n_bytes);
    initialize_memory_pointer_chase(data_factory.data, n_bytes);
    //invalidate_all(data_factory.data, n_bytes);
    dispatch_command(TID, WRITE, data_factory.data, n_bytes);
    double time;
    TIME_FUNCTION_EXECUTION(time, (cacheline_write_function), data_factory.data, size);

    return time;
}

template <int TID>
double contiguous_host_modified_host_read_benchmark(size_t n_bytes) {
    size_t size = n_bytes;
    MallocDataFactory data_factory(n_bytes);
    initialize_memory_pointer_chase(data_factory.data, n_bytes);
    //invalidate_all(data_factory.data, n_bytes);
    dispatch_command(TID, WRITE, data_factory.data, n_bytes);
    double time;
    TIME_FUNCTION_EXECUTION(time, (cacheline_read_function), data_factory.data, size);

    return time;
}

template <int TID>
double contiguous_host_shared_host_write_benchmark(size_t n_bytes) {
    static_assert(TID != 0);
    size_t size = n_bytes;
    MallocDataFactory data_factory(n_bytes);
    initialize_memory_pointer_chase(data_factory.data, n_bytes);
    //invalidate_all(data_factory.data, n_bytes);
    dispatch_command(0, READ, data_factory.data, n_bytes);
    dispatch_command(TID, READ, data_factory.data, n_bytes);
    double time;
    TIME_FUNCTION_EXECUTION(time, (cacheline_write_function), data_factory.data, size);

    return time;
}

template <int TID>
double contiguous_host_exclusive_host_write_benchmark(size_t n_bytes) {
    size_t size = n_bytes;
    MallocDataFactory data_factory(n_bytes);
    initialize_memory_pointer_chase(data_factory.data, n_bytes);
    //invalidate_all(data_factory.data, n_bytes);
    dispatch_command(TID, READ, data_factory.data, n_bytes);
    double time;
    TIME_FUNCTION_EXECUTION(time, (cacheline_write_function), data_factory.data, size);

    return time;
}

template <int TID>
double contiguous_host_shared_host_read_benchmark(size_t n_bytes) {
    static_assert(TID != 0);
    size_t size = n_bytes;
    MallocDataFactory data_factory(n_bytes);
    initialize_memory_pointer_chase(data_factory.data, n_bytes);
    //invalidate_all(data_factory.data, n_bytes);
    dispatch_command(0, READ, data_factory.data, n_bytes);
    dispatch_command(TID, READ, data_factory.data, n_bytes);
    double time;
    TIME_FUNCTION_EXECUTION(time, (cacheline_read_function), data_factory.data, size);

    return time;
}

template <int TID>
double contiguous_host_exclusive_host_read_benchmark(size_t n_bytes) {
    static_assert(TID != 0);
    size_t size = n_bytes;
    MallocDataFactory data_factory(n_bytes);
    initialize_memory_pointer_chase(data_factory.data, n_bytes);
    //invalidate_all(data_factory.data, n_bytes);
    dispatch_command(TID, READ, data_factory.data, n_bytes);
    double time;
    TIME_FUNCTION_EXECUTION(time, (cacheline_read_function), data_factory.data, size);

    return time;
}

double contiguous_untouched_host_write_benchmark(size_t n_bytes) {
    size_t size = n_bytes;
    MallocDataFactory data_factory(n_bytes);
    initialize_memory_pointer_chase(data_factory.data, n_bytes);
    double time = time_function_execution(cacheline_write_function, data_factory.data, size);

    return time;
}

double contiguous_invalid_host_write_benchmark(size_t n_bytes) {
    size_t size = n_bytes;
    MallocDataFactory data_factory(n_bytes);
    initialize_memory_pointer_chase(data_factory.data, n_bytes);
    //invalidate_all(data_factory.data, n_bytes);
    double time;
    TIME_FUNCTION_EXECUTION(time, (cacheline_write_function), data_factory.data, size);

    return time;
}

double contiguous_invalid_host_read_benchmark(size_t n_bytes) {
    size_t size = n_bytes;
    MallocDataFactory data_factory(n_bytes);
    initialize_memory_pointer_chase(data_factory.data, n_bytes);
    //invalidate_all(data_factory.data, n_bytes);
    double time;
    TIME_FUNCTION_EXECUTION(time, (cacheline_read_function), data_factory.data, size);

    return time;
}
