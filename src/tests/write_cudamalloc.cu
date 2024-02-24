int main() {
    unsigned char *ptr;
    cudaMalloc(&ptr, 1024);
    for (size_t i = 0; i < 1024; ++i) {
        ptr[i] = i;
    }
}