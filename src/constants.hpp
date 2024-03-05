#pragma once

#include <unistd.h>

constexpr size_t CPU_CORES = 72;

// ----- smallest power of two that exceeds cache size -----
constexpr size_t CPU_L1_CACHE = 65536;
constexpr size_t CPU_L2_CACHE = 1048576;
constexpr size_t CPU_L3_CACHE = 119537664;

constexpr size_t GPU_L1_CACHE = 2 << 18;
constexpr size_t GPU_L2_CACHE = 52428800;
// ----------

// constexpr size_t PAGE_SIZE = 4096;
constexpr size_t PAGE_SIZE = 1UL << 16;
constexpr size_t CACHELINE_SIZE = 64;
constexpr size_t CACHELINES_PER_PAGE = PAGE_SIZE / CACHELINE_SIZE;
