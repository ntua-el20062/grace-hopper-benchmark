#include <cuda_runtime_api.h>
#include <stdio.h>
#include <fcntl.h>
#include <numa.h>
#include <stdlib.h>
#include <unistd.h>
#include <numaif.h>
#include <cassert>

extern "C" {

void* my_malloc(ssize_t size, int device, cudaStream_t stream) {
   int node = device*8 + 4;
   if (size == 0) {
      return nullptr;
   } else if (size < 256) {
      return malloc(size);
   }

   return numa_alloc_onnode(size, node);
}

void my_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
   // int node = device*8 + 4;
   // printf("freeing %lu bytes on node %d\n", size, device);

   if (size > 0) {
      if (size < 256) {
         free(ptr);
      } else {
         numa_free(ptr, size);
      }
   }
}

}