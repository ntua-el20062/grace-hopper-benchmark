#include <cuda_runtime_api.h>
#include <stdio.h>
#include <fcntl.h>
#include <numa.h>
#include <stdlib.h>
#include <unistd.h>
#include <numaif.h>
#include <cassert>
#include <mutex>

size_t THRESHOLD = 1UL << 20;

size_t normal_max = 0;
size_t normal_cur = 0;
size_t n_normal = 0;
size_t numa_max = 0;
size_t numa_cur = 0;
size_t n_numa = 0;

void print_cur() {
   printf("%lu,%lu,%lu,%lu\n", normal_cur, n_normal, numa_cur, n_numa);
   fflush(stdout);
}

extern "C" {

void* my_malloc(ssize_t size, int device, cudaStream_t stream) {
   void *ptr;
   if (size < THRESHOLD) {
      cudaMallocAsync(&ptr, size, stream);
#ifdef TRACK
      normal_cur += size;
      n_normal++;
      if (normal_cur > normal_max) {
         print_cur();
         normal_max = normal_cur;
      }
#endif
   } else {
      ptr = numa_alloc_onnode(size, NODE);
#ifdef TRACK
      numa_cur += size;
      n_numa++;
      if (numa_cur > numa_max) {
         print_cur();
         numa_max = numa_cur;
      }
#endif
   }

   return ptr;
}

void my_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
   if (size < THRESHOLD) {
      cudaFreeAsync(ptr, stream);
#ifdef TRACK
      normal_cur -= size;
      n_normal--;
#endif
   } else {
      numa_free(ptr, size);
#ifdef TRACK
      numa_cur -= size;
      n_numa--;
#endif
   }
}

void* cuda_malloc(ssize_t size, int device, cudaStream_t stream) {
   void *ptr;
   cudaMallocAsync(&ptr, size, stream);

   return ptr;
}

void cuda_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
   cudaFreeAsync(ptr, stream);
}

}