#pragma once

#include "thread_tools.hpp"
#include "memory.hpp"

__global__ void latency_kernel(uint8_t *in, size_t n_elem, size_t n_iter, double *time) {
   
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
        printf("start latency kernel\n");
    printf("num iterations: %lu\n", n_iter);

    for (size_t iter = 0; iter < n_iter; ++iter) {
                printf("iter: %lu\n", iter);


        uint8_t *itr = in;
        clock_t start = clock();
        for (size_t i = 0; i < n_elem/8; ++i) {
            itr = *((uint8_t **) itr);
            itr = *((uint8_t **) itr);
            itr = *((uint8_t **) itr);
            itr = *((uint8_t **) itr);

            itr = *((uint8_t **) itr);
            itr = *((uint8_t **) itr);
            itr = *((uint8_t **) itr);
            itr = *((uint8_t **) itr);
        }
        clock_t end = clock();
        time[iter] = end - start;

        if (tid > 1) { // dummy work
            printf("%u ", *itr);
        }

    }
}

__global__ void latency_write_kernel(uint8_t *in, size_t n_elem, size_t n_iter, double *time) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    for (size_t iter = 0; iter < n_iter; ++iter) {
        uint8_t *itr = in;
        clock_t start = clock();
        for (size_t i = 0; i < n_elem/8; ++i) {
            itr = *((uint8_t **) itr);
            *((size_t *) itr + 1) = n_iter;
            itr = *((uint8_t **) itr);
            *((size_t *) itr + 1) = n_iter;
            itr = *((uint8_t **) itr);
            *((size_t *) itr + 1) = n_iter;
            itr = *((uint8_t **) itr);
            *((size_t *) itr + 1) = n_iter;

            itr = *((uint8_t **) itr);
            *((size_t *) itr + 1) = n_iter;
            itr = *((uint8_t **) itr);
            *((size_t *) itr + 1) = n_iter;
            itr = *((uint8_t **) itr);
            *((size_t *) itr + 1) = n_iter;
            itr = *((uint8_t **) itr);
            *((size_t *) itr + 1) = n_iter;
        }
        clock_t end = clock();
        time[iter] = end - start;

        if (tid > 1) { // dummy work
            printf("%u ", *itr);
        }

    }
}

template <bool IS_PAGE, bool IS_WRITE, typename ALLOC>
void latency_test_host_template(size_t n_iter, size_t n_bytes, size_t t_id, std::string name) {
    double times[n_iter];
    ALLOC factory(n_bytes);
    size_t n_elem;
    if constexpr (IS_PAGE) {
        initialize_memory_page_chase(factory.data, n_bytes);
        n_elem = n_bytes / PAGE_SIZE;
    } else {
        initialize_memory_pointer_chase(factory.data, n_bytes);
        n_elem = n_bytes / (CACHELINE_SIZE*2);
    }

    if constexpr (IS_WRITE) {
        dispatch_command(t_id, LATENCY_WRITE_TEST, factory.data, n_elem, (uint8_t *) times, n_iter);
    } else {
        dispatch_command(t_id, LATENCY_TEST, factory.data, n_elem, (uint8_t *) times, n_iter);
    }

    // for (size_t i = 0; i < n_iter; ++i) {
    //     if constexpr (IS_WRITE) {
    //         times[i] = latency_write_function(factory.data, n_elem); 
    //     } else {
    //         times[i] = latency_function(factory.data, n_elem);
    //     }
    //     prepare_memory(READ, cache_filler, sizeof(cache_filler));
    // }

    millisecond_times_to_latency_ns_file(times, n_iter, n_elem, name);
}

template <bool IS_PAGE>
void run_latency_test_host(size_t n_iter, size_t n_bytes) {
	std::cout << "1" << std::endl;
	std::string base = "results/latency/";
	        std::cout << "2" << std::endl;

    if (!IS_PAGE) {
        base += "fine/";
    }

    std::cout << "3" << std::endl;

    latency_test_host_template<IS_PAGE, false, HOST_MEM>(n_iter, n_bytes, 0, base + "host/ddr");
    std::cout << "4" << std::endl;

    latency_test_host_template<IS_PAGE, false, DEVICE_MEM>(n_iter, n_bytes, 0, base + "host/hbm");
    std::cout << "5" << std::endl;

    // latency_test_host_template<IS_PAGE, false, REMOTE_HOST_MEM>(n_iter, n_bytes, 0, base + "host/ddr_remote");
    //latency_test_host_template<IS_PAGE, false, REMOTE_DEVICE_MEM>(n_iter, n_bytes, 0, base + "host/hbm_remote");
    // latency_test_host_template<IS_PAGE, false, FAR_HOST_MEM>(n_iter, n_bytes, 72, base + "host/ddr_far");
    // latency_test_host_template<IS_PAGE, false, FAR_DEVICE_MEM>(n_iter, n_bytes, 72, base + "host/hbm_far");
}

template <bool IS_PAGE, bool IS_WRITE, typename ALLOC>
void latency_test_device_template(size_t n_iter, size_t n_bytes, std::string name) {
    double gpu_clock = get_gpu_clock_khz();
    double times[n_iter];
    ALLOC factory(n_bytes);
    size_t n_elem;
    if constexpr (IS_PAGE) {
        initialize_memory_page_chase(factory.data, n_bytes);
        n_elem = n_bytes / PAGE_SIZE;
    } else {
        initialize_memory_pointer_chase(factory.data, n_bytes);
        n_elem = n_bytes / (CACHELINE_SIZE*2);
    }
    if constexpr (IS_WRITE) {
        latency_write_kernel<<<1, 1>>>(factory.data, n_elem, n_iter, times);
    } else {
        std::cout << "before latency kernel" << std::endl;
        latency_kernel<<<1, 1>>>(factory.data, n_elem, n_iter, times);
    }
    std::cout << "before sync" << std::endl;
    cudaDeviceSynchronize();
    std::cout << "after sync" << std::endl;
    for (size_t i = 0; i < n_iter; ++i) {
        times[i] /= gpu_clock;
    }
    millisecond_times_to_latency_ns_file(times, n_iter, n_elem, name);
    std::cout << "DONE INSIDE THE TEMPLATE" << std::endl;
}

template <bool IS_PAGE>
void run_latency_test_device(size_t n_iter, size_t n_bytes) {
    std::string base = "results/latency/";
    std::cout << "1" << std::endl;
    if (!IS_PAGE) {
        base += "fine/";
    }
    std::cout << "2" << std::endl;
    std::cout << "start  run_latency_test_device" << std::endl;
    std::cout << "3" << std::endl;
    std::cout <<"start latency devide to host" << std::endl;
    latency_test_device_template<IS_PAGE, false, HOST_MEM>(n_iter, n_bytes, base + "device/ddr");
    std::cout <<"start latency devide to device" << std::endl;
    latency_test_device_template<IS_PAGE, false, DEVICE_MEM>(n_iter, n_bytes, base + "device/hbm");

    std::cout << "done with  run_latency_test_device " << std::endl;
    //latency_test_device_template<IS_PAGE, false, REMOTE_HOST_MEM>(n_iter, n_bytes, base + "device/ddr_remote");
    //latency_test_device_template<IS_PAGE, false, REMOTE_DEVICE_MEM>(n_iter, n_bytes, base + "device/hbm_remote");
}
