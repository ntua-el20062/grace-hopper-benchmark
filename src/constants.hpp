#pragma once

constexpr size_t CPU_CORES = 72;

// ----- smallest power of two that exceeds cache size -----
constexpr size_t CPU_L1_CACHE = 64000;
constexpr size_t CPU_L2_CACHE = 1000000;
constexpr size_t CPU_L3_CACHE = 114000000;

constexpr size_t GPU_L1_CACHE = 2 << 18;
constexpr size_t GPU_L2_CACHE = 52428800;
// ----------

constexpr size_t PAGE_SIZE = 1 << 12;
constexpr size_t CACHELINE_SIZE = 64;