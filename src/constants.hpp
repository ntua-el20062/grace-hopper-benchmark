constexpr size_t CPU_CORES = 72;

// ----- smallest power of two that exceeds cache size -----
constexpr size_t CPU_L1_CACHE = 1 << 16;
constexpr size_t CPU_L2_CACHE = 1 << 20;
constexpr size_t CPU_L3_CACHE = 1 << 27;

constexpr size_t GPU_L1_CACHE = 1 << 18;
constexpr size_t GPU_L2_CACHE = 1 << 26;
// ----------

constexpr size_t PAGE_SIZE = 1 << 12;
constexpr size_t CACHELINE_SIZE = 64;