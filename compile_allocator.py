import os
import sys

additional_flags = ' '.join(sys.argv[1:])

print('compiling with additional flags: ', additional_flags)

nodes = [0, 1]

for n in nodes:
    os.system(f'gcc -fPIC -shared -DNODE={n} {additional_flags} -I$CUDA_HOME/include alloc.cpp -o alloc{n}.so -lnuma')
