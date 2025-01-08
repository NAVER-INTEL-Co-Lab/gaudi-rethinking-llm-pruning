import torch
from transformers import LlamaForCausalLM as Llama_Huggingface
import torch
import habana_frameworks.torch.core as htcore

model_name = 'meta-llama/Llama-2-7b-hf' 
model = Llama_Huggingface.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=None, 
        low_cpu_mem_usage=True, 
        device_map="hpu"
    )

print(model)