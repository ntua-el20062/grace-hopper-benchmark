#include "base_functions.hpp"

template <int TID>
float contiguous_host_modified_host_write_benchmark(size_t n_bytes) {
    size_t size = n_bytes / sizeof(double);
    MallocDataFactory<HostModifiedInit<TID>> data_factory(n_bytes);
    float time = time_function_execution(strided_write_function<double, 1>, (double *) data_factory.data, size);

    return time;
}

float contiguous_untouched_host_copy_benchmark(size_t n_bytes) {
    size_t size = n_bytes / sizeof(double);
    MallocDataFactory<CacheFlushInit> out_data_factory(n_bytes);
    MallocDataFactory<CacheFlushInit> in_data_factory(n_bytes);
    float time = time_function_execution(strided_copy_function<double, 1>, (double *) out_data_factory.data, (double *) in_data_factory.data, size);

    return time;
}

float contiguous_host_exclusive_host_copy_benchmark(size_t n_bytes) {
    size_t size = n_bytes / sizeof(double);
    MallocDataFactory<HostModifiedInit<>> out_data_factory(n_bytes);
    MallocDataFactory<HostModifiedInit<>> in_data_factory(n_bytes);
    float time = time_function_execution(strided_copy_function<double, 1>, (double *) out_data_factory.data, (double *) in_data_factory.data, size);

    return time;
}

float contiguous_untouched_host_write_benchmark(size_t n_bytes) {
    size_t size = n_bytes / sizeof(double);
    MallocDataFactory<NoopInit> data_factory(n_bytes);
    float time = time_function_execution(strided_write_function<double, 1>, (double *) data_factory.data, size);

    return time;
}

float contiguous_invalid_host_write_benchmark(size_t n_bytes) {
    size_t size = n_bytes / sizeof(double);
    MallocDataFactory<CacheFlushInit> data_factory(n_bytes);
    float time = time_function_execution(strided_write_function<double, 1>, (double *) data_factory.data, size);

    return time;
}
