import torch
new_alloc = torch.cuda.memory.CUDAPluggableAllocator('/users/lfusco/code/gh_benchmark/alloc.so', 'my_malloc', 'my_free')
torch.cuda.memory.change_current_allocator(new_alloc)

from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers import GPT2Tokenizer
from datasets import load_dataset

opt_families = [
    'facebook/opt-125m',
    'facebook/opt-13b',
    'facebook/opt-30b',
    'facebook/opt-66b',
]
llama = 'meta-llama/Llama-2-70b-chat-hf'

class Evaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in self.dataset:
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc
    
tokenizer = GPT2Tokenizer.from_pretrained(opt_families[-2])
dataset = load_dataset('lambada', split='validation[:1000]')
evaluator = Evaluator(dataset, tokenizer, 'cuda')


model_fp16 = OPTForCausalLM.from_pretrained(opt_families[-2], torch_dtype=torch.float16).cuda()
acc_fp16 = evaluator.evaluate(model_fp16)
print(f'Original model (fp16) accuracy: {acc_fp16}')
