import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_name_or_path = "llama-3.1-8B-Instruct"
lora_path = "output/lora_finetuned_llama-3.1-8B-Instruct"
output_path = "merged_lora_llama"
print(f"Loading the base model from {model_name_or_path}")
base = AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
)
base_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

print(f"Loading the LoRA adapter from {lora_path}")

lora_model = PeftModel.from_pretrained(
    base,
    lora_path,
    torch_dtype=torch.float16,
)

print("Applying the LoRA")
model = lora_model.merge_and_unload()

print(f"Saving the target model to {output_path}")
model.save_pretrained(output_path)
base_tokenizer.save_pretrained(output_path)