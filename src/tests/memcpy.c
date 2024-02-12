#include <string.h>

#define SIZE 1UL << 16

int main() {
    char a[SIZE];
    char b[SIZE];

    memset(a, 0xff, SIZE);
    memcpy(b, a, SIZE);

    return 0;
}