#include <nccl.h>
#include <mpi.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <string>
#include <cassert>
#include <fcntl.h>
#include <numa.h>
#include <stdlib.h>
#include <unistd.h>
#include <numaif.h>
#include <sched.h>
// #include "memory.hpp"
#include "measurement.hpp"

enum ALLOC_POLICY {
    CUR_GPU,
    OTHER_GPU,
    CUR_CPU,
    OTHER_CPU
};

MPI_Comm local_comm;

__attribute__((always_inline)) inline uint64_t my_barrier(MPI_Comm &comm) {
    uint64_t clock;
    int rank;
    MPI_Comm_rank(local_comm, &rank);
    if (rank == 0) {
        clock = get_cpu_clock() + 1000 * 1000;
    }
    MPI_Bcast(&clock, 1, MPI_UNSIGNED_LONG, 0, comm);

    uint64_t check;
    while ((check = get_cpu_clock()) < clock);

    return check;
}

__attribute__((always_inline)) inline uint64_t measure_end_time(MPI_Comm &comm) {
    uint64_t clock = get_cpu_clock();

    MPI_Allreduce(MPI_IN_PLACE, &clock, 1, MPI_UNSIGNED_LONG, MPI_MAX, comm);

    return clock;
}

int get_node_from_policy(int device, ALLOC_POLICY policy) {
    switch (policy) {
        case CUR_GPU:
            return device*8 + 4;
        case OTHER_GPU:
            return ((device+2)%4)*8 + 4;
        case CUR_CPU:
            return device;
        case OTHER_CPU:
            return (device+2)%4;
    }

    return -1;
}

int get_device_from_policy(int device, ALLOC_POLICY policy) {
    switch (policy) {
        case CUR_GPU:
            return device;
        case OTHER_GPU:
            return (device+2)%4;
    }

    return -1;
}

template <ALLOC_POLICY POLICY>
struct CUDA_ALLOC {
    uint8_t *data = nullptr;
    size_t size = 0;

    CUDA_ALLOC(size_t n_bytes) {
        int cur_device;
        cudaGetDevice(&cur_device);
        
        int target_device = get_device_from_policy(cur_device, POLICY);
        cudaSetDevice(target_device);
        cudaMalloc(&data, n_bytes);
        cudaMemset(data, 0xff, n_bytes);
        cudaDeviceSynchronize();
        cudaSetDevice(cur_device);
        size = n_bytes;
    }

    ~CUDA_ALLOC() {
        cudaFree(data);
    }
};

template <ALLOC_POLICY POLICY>
struct NUMA_ALLOC {
    uint8_t *data = nullptr;
    size_t size = 0;

    NUMA_ALLOC(size_t n_bytes) {
        int device;
        cudaGetDevice(&device);
        
        data = (uint8_t *) numa_alloc_onnode(n_bytes, get_node_from_policy(device, POLICY));
        memset(data, 0xff, n_bytes);
        size = n_bytes;
    }

    ~NUMA_ALLOC() {
        numa_free(data, size);
    }
};

template <typename SRC_ALLOC, typename DST_ALLOC, bool BI>
void send_recv(int rank, int world_size, size_t n_iter, size_t n_bytes, ncclComm_t &comm, cudaStream_t &stream, std::string name) {
    float times[n_iter];

    SRC_ALLOC src(n_bytes);
    DST_ALLOC dst(n_bytes);

    MPI_Barrier(MPI_COMM_WORLD);

    for (size_t iter = 0; iter < n_iter; ++iter) {
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        cudaEventRecord(start, stream);

        ncclGroupStart();
        if constexpr (BI) {
            ncclSend(src.data, n_bytes, ncclChar, (rank+1)%2, comm, stream);
            ncclRecv(dst.data, n_bytes, ncclChar, (rank+1)%2, comm, stream);
        } else {
            if (rank == 0) {
                ncclSend(src.data, n_bytes, ncclChar, 1, comm, stream);
            } else {
                ncclRecv(dst.data, n_bytes, ncclChar, 0, comm, stream);
            }
        }
        ncclGroupEnd();

        cudaEventRecord(end, stream);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&times[iter], start, end);
        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }

    if (rank == 0) {
        if constexpr (BI) {
            millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/nccl/send_recv_bi/" + name);
        } else {
            millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/nccl/send_recv/" + name);
        }
    }
}

template <typename ALLOC>
void all_reduce(int rank, int world_size, size_t n_iter, size_t n_bytes, ncclComm_t &comm, cudaStream_t &stream, std::string name) {
    if (n_bytes % sizeof(double) != 0) {
        return;
    }

    float times[n_iter];

    ALLOC data(n_bytes);

    MPI_Barrier(MPI_COMM_WORLD);

    for (size_t iter = 0; iter < n_iter; ++iter) {
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        cudaEventRecord(start, stream);

        ncclAllReduce(data.data, data.data, n_bytes / sizeof(double), ncclDouble, ncclSum, comm, stream);

        cudaEventRecord(end, stream);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&times[iter], start, end);
        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }

    if (rank == 0) {
        millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/nccl/all_reduce/" + name);
    }
}

template <typename ALLOC>
void all_gather(int rank, int world_size, size_t n_iter, size_t n_bytes, ncclComm_t &comm, cudaStream_t &stream, std::string name) {
    float times[n_iter];

    ALLOC data(n_bytes * world_size);

    MPI_Barrier(MPI_COMM_WORLD);

    for (size_t iter = 0; iter < n_iter; ++iter) {
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        cudaEventRecord(start, stream);

        ncclAllGather(data.data + n_bytes * rank, data.data, n_bytes, ncclChar, comm, stream);

        cudaEventRecord(end, stream);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&times[iter], start, end);
        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }

    if (rank == 0) {
        millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/nccl/all_gather/" + name);
    }
}

void run_all_gather_tests(int rank, int world_size, int proc_per_node, size_t n_iter, size_t n_bytes, ncclComm_t &comm, cudaStream_t &stream) {
    std::string m, tail;
    if (world_size > proc_per_node) {
        m = "remote/";
        tail = std::to_string(world_size / proc_per_node);
    } else {
        m = "local/";
        tail = std::to_string(n_bytes);
    }
    all_gather<NUMA_ALLOC<CUR_CPU>>(rank, world_size, n_iter, n_bytes, comm, stream, m + "ddr/" + tail);
    all_gather<NUMA_ALLOC<CUR_GPU>>(rank, world_size, n_iter, n_bytes, comm, stream, m + "hbm/" + tail);
    all_gather<NUMA_ALLOC<OTHER_CPU>>(rank, world_size, n_iter, n_bytes, comm, stream, m + "ddr_remote/" + tail);
    all_gather<NUMA_ALLOC<OTHER_GPU>>(rank, world_size, n_iter, n_bytes, comm, stream, m + "hbm_remote/" + tail);
    // all_gather<CUDA_ALLOC<CUR_GPU>>(rank, world_size, n_iter, n_bytes, comm, stream, m + "cuda/" + tail);
}

void run_all_reduce_tests(int rank, int world_size, int proc_per_node, size_t n_iter, size_t n_bytes, ncclComm_t &comm, cudaStream_t &stream) {
    std::string m, tail;
    if (world_size > proc_per_node) {
        m = "remote/";
        tail = std::to_string(world_size / proc_per_node);
    } else {
        m = "local/";
        tail = std::to_string(n_bytes);
    }
    all_reduce<NUMA_ALLOC<CUR_CPU>>(rank, world_size, n_iter, n_bytes, comm, stream, m + "ddr/" + tail);
    all_reduce<NUMA_ALLOC<CUR_GPU>>(rank, world_size, n_iter, n_bytes, comm, stream, m + "hbm/" + tail);
    all_reduce<NUMA_ALLOC<OTHER_CPU>>(rank, world_size, n_iter, n_bytes, comm, stream, m + "ddr_remote/" + tail);
    all_reduce<NUMA_ALLOC<OTHER_GPU>>(rank, world_size, n_iter, n_bytes, comm, stream, m + "hbm_remote/" + tail);
    // all_reduce<CUDA_ALLOC<CUR_GPU>>(rank, world_size, n_iter, n_bytes, comm, stream, m + "cuda/" + tail);
}

template <bool BI>
void run_send_recv_tests(int rank, int world_size, int proc_per_node, size_t n_iter, size_t n_bytes, ncclComm_t &comm, cudaStream_t &stream) {
    assert(world_size == 2);
    std::string m;
    if (proc_per_node == 2) {
        m = "local/";
    } else if (proc_per_node == 1) {
        m = "remote/";
    }
    if (rank == 0) {
        printf("Running send_recv %s %s with %lu bytes\n", BI ? "bidirectional" : "single", proc_per_node == 2 ? "local" : "remote", n_bytes);
        fflush(stdout);
    }
    send_recv<NUMA_ALLOC<CUR_CPU>, NUMA_ALLOC<CUR_CPU>, BI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "ddr_ddr/" + std::to_string(n_bytes));
    send_recv<NUMA_ALLOC<CUR_CPU>, NUMA_ALLOC<CUR_GPU>, BI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "ddr_hbm/" + std::to_string(n_bytes));
    send_recv<NUMA_ALLOC<CUR_CPU>, NUMA_ALLOC<OTHER_CPU>, BI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "ddr_ddr_remote/" + std::to_string(n_bytes));
    send_recv<NUMA_ALLOC<CUR_CPU>, NUMA_ALLOC<OTHER_GPU>, BI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "ddr_hbm_remote/" + std::to_string(n_bytes));
    send_recv<NUMA_ALLOC<CUR_GPU>, NUMA_ALLOC<CUR_CPU>, BI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "hbm_ddr/" + std::to_string(n_bytes));
    send_recv<NUMA_ALLOC<CUR_GPU>, NUMA_ALLOC<CUR_GPU>, BI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "hbm_hbm/" + std::to_string(n_bytes));
    send_recv<NUMA_ALLOC<CUR_GPU>, NUMA_ALLOC<OTHER_CPU>, BI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "hbm_ddr_remote/" + std::to_string(n_bytes));
    send_recv<NUMA_ALLOC<CUR_GPU>, NUMA_ALLOC<OTHER_GPU>, BI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "hbm_hbm_remote/" + std::to_string(n_bytes));
    send_recv<NUMA_ALLOC<OTHER_CPU>, NUMA_ALLOC<CUR_CPU>, BI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "ddr_remote_ddr/" + std::to_string(n_bytes));
    send_recv<NUMA_ALLOC<OTHER_CPU>, NUMA_ALLOC<CUR_GPU>, BI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "ddr_remote_hbm/" + std::to_string(n_bytes));
    send_recv<NUMA_ALLOC<OTHER_CPU>, NUMA_ALLOC<OTHER_CPU>, BI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "ddr_remote_ddr_remote/" + std::to_string(n_bytes));
    send_recv<NUMA_ALLOC<OTHER_CPU>, NUMA_ALLOC<OTHER_GPU>, BI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "ddr_remote_hbm_remote/" + std::to_string(n_bytes));
    send_recv<NUMA_ALLOC<OTHER_GPU>, NUMA_ALLOC<CUR_CPU>, BI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "hbm_remote_ddr/" + std::to_string(n_bytes));
    send_recv<NUMA_ALLOC<OTHER_GPU>, NUMA_ALLOC<CUR_GPU>, BI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "hbm_remote_hbm/" + std::to_string(n_bytes));
    send_recv<NUMA_ALLOC<OTHER_GPU>, NUMA_ALLOC<OTHER_CPU>, BI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "hbm_remote_ddr_remote/" + std::to_string(n_bytes));
    send_recv<NUMA_ALLOC<OTHER_GPU>, NUMA_ALLOC<OTHER_GPU>, BI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "hbm_remote_hbm_remote/" + std::to_string(n_bytes));
}

template <typename SRC_ALLOC, typename DST_ALLOC, bool BI, bool MPI>
void group_send_recv(int rank, int world_size, size_t n_iter, size_t n_bytes, ncclComm_t &comm, cudaStream_t &stream, std::string name) {
    int half_world = world_size / 2;
    double times[n_iter];

    SRC_ALLOC src(n_bytes);
    DST_ALLOC dst(n_bytes);

    MPI_Request requests[2*world_size];

    for (size_t iter = 0; iter < n_iter; ++iter) {
        MPI_Barrier(MPI_COMM_WORLD);

        uint64_t a = get_cpu_clock();

        if (rank < half_world) {
            a = my_barrier(local_comm);
        }

        if constexpr (!MPI)
            ncclGroupStart();

        if (rank < half_world) {
            if constexpr (MPI) {
                MPI_Isend(src.data, n_bytes / sizeof(double), MPI_DOUBLE, rank + half_world, 0, MPI_COMM_WORLD, &requests[rank]);
                if constexpr (BI)
                    MPI_Irecv(dst.data, n_bytes / sizeof(double), MPI_DOUBLE, rank + half_world, 0, MPI_COMM_WORLD, &requests[rank+world_size]);
            } else {
                ncclSend(src.data, n_bytes, ncclChar, rank + half_world, comm, stream);
                if constexpr (BI)
                    ncclRecv(dst.data, n_bytes, ncclChar, rank + half_world, comm, stream);
            }
        } else {
            if constexpr (MPI) {
                if constexpr (BI)
                    MPI_Isend(src.data, n_bytes / sizeof(double), MPI_DOUBLE, rank - half_world, 0, MPI_COMM_WORLD, &requests[rank+world_size]);
                MPI_Irecv(dst.data, n_bytes / sizeof(double), MPI_DOUBLE, rank - half_world, 0, MPI_COMM_WORLD, &requests[rank]);
            } else {
                if constexpr (BI)
                    ncclSend(src.data, n_bytes, ncclChar, rank - half_world, comm, stream);
                ncclRecv(dst.data, n_bytes, ncclChar, rank - half_world, comm, stream);
            }
        }

        if constexpr (MPI) {
            MPI_Wait(&requests[rank], MPI_STATUS_IGNORE);
            if constexpr (BI)
                MPI_Wait(&requests[rank+world_size], MPI_STATUS_IGNORE);
        } else {
            ncclGroupEnd();
            cudaStreamSynchronize(stream);
            cudaDeviceSynchronize();
        }

        uint64_t b = measure_end_time(local_comm);
        times[iter] = get_elapsed_milliseconds_clock(a, b);
    }

    if (rank == 0) {
        int multiplier = BI ? 2 : 1;
        millisecond_times_to_gb_sec_file(times, n_iter, n_bytes * half_world * multiplier, "results/nccl/group_send_recv/" + name);
    }
}

template <bool BI, bool MPI>
void run_group_send_recv_tests(int rank, int world_size, int proc_per_node, size_t n_iter, size_t n_bytes, ncclComm_t &comm, cudaStream_t &stream) {
    std::string m = std::string(BI ? "bi/" : "mono/") + std::string(MPI ? "mpi/" : "nccl/") + std::to_string(proc_per_node) + "/";
    if (rank == 0) {
        printf("Running group_send_recv %s with %lu bytes\n", m.c_str(), n_bytes);
        fflush(stdout);
    }
    group_send_recv<NUMA_ALLOC<CUR_CPU>, NUMA_ALLOC<CUR_CPU>, BI, MPI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "ddr_ddr/" + std::to_string(n_bytes));
    group_send_recv<NUMA_ALLOC<CUR_CPU>, NUMA_ALLOC<CUR_GPU>, BI, MPI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "ddr_hbm/" + std::to_string(n_bytes));
    group_send_recv<NUMA_ALLOC<CUR_CPU>, NUMA_ALLOC<OTHER_CPU>, BI, MPI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "ddr_ddr_remote/" + std::to_string(n_bytes));
    group_send_recv<NUMA_ALLOC<CUR_CPU>, NUMA_ALLOC<OTHER_GPU>, BI, MPI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "ddr_hbm_remote/" + std::to_string(n_bytes));
    group_send_recv<NUMA_ALLOC<CUR_CPU>, CUDA_ALLOC<CUR_GPU>, BI, MPI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "ddr_cuda/" + std::to_string(n_bytes));
    group_send_recv<NUMA_ALLOC<CUR_GPU>, NUMA_ALLOC<CUR_CPU>, BI, MPI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "hbm_ddr/" + std::to_string(n_bytes));
    group_send_recv<NUMA_ALLOC<CUR_GPU>, NUMA_ALLOC<CUR_GPU>, BI, MPI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "hbm_hbm/" + std::to_string(n_bytes));
    group_send_recv<NUMA_ALLOC<CUR_GPU>, NUMA_ALLOC<OTHER_CPU>, BI, MPI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "hbm_ddr_remote/" + std::to_string(n_bytes));
    group_send_recv<NUMA_ALLOC<CUR_GPU>, NUMA_ALLOC<OTHER_GPU>, BI, MPI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "hbm_hbm_remote/" + std::to_string(n_bytes));
    group_send_recv<NUMA_ALLOC<CUR_GPU>, CUDA_ALLOC<CUR_GPU>, BI, MPI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "hbm_cuda/" + std::to_string(n_bytes));
    group_send_recv<NUMA_ALLOC<OTHER_CPU>, NUMA_ALLOC<CUR_CPU>, BI, MPI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "ddr_remote_ddr/" + std::to_string(n_bytes));
    group_send_recv<NUMA_ALLOC<OTHER_CPU>, NUMA_ALLOC<CUR_GPU>, BI, MPI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "ddr_remote_hbm/" + std::to_string(n_bytes));
    group_send_recv<NUMA_ALLOC<OTHER_CPU>, NUMA_ALLOC<OTHER_CPU>, BI, MPI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "ddr_remote_ddr_remote/" + std::to_string(n_bytes));
    group_send_recv<NUMA_ALLOC<OTHER_CPU>, NUMA_ALLOC<OTHER_GPU>, BI, MPI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "ddr_remote_hbm_remote/" + std::to_string(n_bytes));
    group_send_recv<NUMA_ALLOC<OTHER_CPU>, CUDA_ALLOC<CUR_GPU>, BI, MPI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "ddr_remote_cuda/" + std::to_string(n_bytes));
    group_send_recv<NUMA_ALLOC<OTHER_GPU>, NUMA_ALLOC<CUR_CPU>, BI, MPI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "hbm_remote_ddr/" + std::to_string(n_bytes));
    group_send_recv<NUMA_ALLOC<OTHER_GPU>, NUMA_ALLOC<CUR_GPU>, BI, MPI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "hbm_remote_hbm/" + std::to_string(n_bytes));
    group_send_recv<NUMA_ALLOC<OTHER_GPU>, NUMA_ALLOC<OTHER_CPU>, BI, MPI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "hbm_remote_ddr_remote/" + std::to_string(n_bytes));
    group_send_recv<NUMA_ALLOC<OTHER_GPU>, NUMA_ALLOC<OTHER_GPU>, BI, MPI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "hbm_remote_hbm_remote/" + std::to_string(n_bytes));
    group_send_recv<NUMA_ALLOC<OTHER_GPU>, CUDA_ALLOC<CUR_GPU>, BI, MPI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "hbm_remote_cuda/" + std::to_string(n_bytes));
    group_send_recv<CUDA_ALLOC<CUR_GPU>, NUMA_ALLOC<CUR_CPU>, BI, MPI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "cuda_ddr/" + std::to_string(n_bytes));
    group_send_recv<CUDA_ALLOC<CUR_GPU>, NUMA_ALLOC<CUR_GPU>, BI, MPI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "cuda_hbm/" + std::to_string(n_bytes));
    group_send_recv<CUDA_ALLOC<CUR_GPU>, NUMA_ALLOC<OTHER_CPU>, BI, MPI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "cuda_ddr_remote/" + std::to_string(n_bytes));
    group_send_recv<CUDA_ALLOC<CUR_GPU>, NUMA_ALLOC<OTHER_GPU>, BI, MPI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "cuda_hbm_remote/" + std::to_string(n_bytes));
    group_send_recv<CUDA_ALLOC<CUR_GPU>, CUDA_ALLOC<CUR_GPU>, BI, MPI>(rank, world_size, n_iter, n_bytes, comm, stream, m + "cuda_cuda/" + std::to_string(n_bytes));


}

static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

int main(int argc, char *argv[]) {
    MPI_Init(nullptr, nullptr);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int gpu_id = 0;
    int proc_per_node = 0;
    //calculating localRank based on hostname which is used in selecting a GPU
    uint64_t hostHashs[world_size];
    char hostname[1024];
    getHostName(hostname, 1024);
    uint64_t my_hash = getHostHash(hostname);
    hostHashs[rank] = my_hash;
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);
    for (int p=0; p<world_size; p++) {
        if (hostHashs[p] == hostHashs[rank]) {
            if (p < rank) {
                gpu_id++;
            }
            proc_per_node++;
        }
    }

    MPI_Comm_split(MPI_COMM_WORLD, my_hash, rank, &local_comm);
    int local_rank, local_size;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);

    printf("HOST %s, GPU %d/%d, GLOBAL %d/%d LOCAL %d/%d, TID %d\n", hostname, gpu_id, proc_per_node, rank, world_size, local_rank, local_size, sched_getcpu());
    printf("%lu\n", my_barrier(local_comm));
    fflush(stdout);

    cudaSetDevice(gpu_id);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t comm;
    ncclCommInitRank(&comm, world_size, id, rank);

    for (size_t n_bytes = 1UL << 33; n_bytes <= 1UL << 33; n_bytes *= 2) {
        std::cout << n_bytes << std::endl;
        // run_send_recv_tests<true>(rank, world_size, proc_per_node, 100, n_bytes, comm, stream);
        // run_send_recv_tests<false>(rank, world_size, proc_per_node, 100, n_bytes, comm, stream);
        // run_all_reduce_tests(rank, world_size, proc_per_node, 10, n_bytes, comm, stream);
        // run_all_gather_tests(rank, world_size, proc_per_node, 10, n_bytes, comm, stream);
        // MPI
        run_group_send_recv_tests<true, true>(rank, world_size, proc_per_node, 10, n_bytes, comm, stream);
        run_group_send_recv_tests<false, true>(rank, world_size, proc_per_node, 10, n_bytes, comm, stream);
        // NCCL
        // run_group_send_recv_tests<true, false>(rank, world_size, proc_per_node, 10, n_bytes, comm, stream);
        // run_group_send_recv_tests<false, false>(rank, world_size, proc_per_node, 10, n_bytes, comm, stream);
    }

    run_all_reduce_tests(rank, world_size, proc_per_node, 10, 1UL << 32, comm, stream);
    run_all_gather_tests(rank, world_size, proc_per_node, 10, 1UL << 24, comm, stream);

    ncclCommDestroy(comm);

    cudaStreamDestroy(stream);

    MPI_Finalize();
}