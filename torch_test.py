import torch

print(torch.cuda.is_available())

new_alloc = torch.cuda.memory.CUDAPluggableAllocator('/users/lfusco/code/gh_benchmark/alloc.so', 'my_malloc', 'my_free')
torch.cuda.memory.change_current_allocator(new_alloc)


x = torch.empty(100000, 100000, device=torch.device('cuda'))
x[:] = 1
print(x)