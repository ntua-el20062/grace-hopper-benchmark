import torch
import sys

NUMA = int(sys.argv[1])
NITER = int(sys.argv[2])

new_alloc = torch.cuda.memory.CUDAPluggableAllocator(f'/users/lfusco/code/gh_benchmark/alloc{NUMA}.so', 'my_malloc', 'my_free')
torch.cuda.memory.change_current_allocator(new_alloc)

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

import time

access_token = ''
cache_dir = '/bret/scratch/cscs/lfusco/.cache/huggingface'
device = 'cuda:0'

model_17 = "meta-llama/Llama-2-13b-chat-hf"
model_33 = "bofenghuang/vigogne-33b-instruct"
# model_33 = "alexl83/LLaMA-33B-HF"

tokenizer = AutoTokenizer.from_pretrained(model_33, token=access_token, cache_dir=cache_dir, legacy=True)
model = AutoModelForCausalLM.from_pretrained(model_33, token=access_token, cache_dir=cache_dir)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device=device
)

for i in range(NITER):
    start = time.time()
    sequences = pipeline(
        '',
        do_sample=True,
        top_k=1,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=100,
        # temperature=0.0001
    )
    end = time.time()
    print(end - start)
