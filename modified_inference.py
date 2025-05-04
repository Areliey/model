from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from peft import PeftModel, LoraConfig, TaskType
from attn_modify import MyQwen2Attn

device = 'cuda'
mode_path = 'Qwen/Qwen2.5-0.5B-Instruct'
lora_path = '/root/autodl-tmp/Qwen/Qwen_lora'
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(lora_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16)

index = 1
qwen_config = AutoConfig.from_pretrained(mode_path)
for layer in model.model.layers:
    layer.self_attn = MyQwen2Attn(qwen_config, index).to(device)
    index = index + 1

# 加载lora权重
# model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

model = PeftModel(model, config)
model.add_adapter(lora_path, config)
print(model)

prompt = "能给我讲一个寓意深刻的故事吗？"
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512,
    do_sample=True,
    top_p=0.9, 
    temperature=0.5, 
    repetition_penalty=1.1,
    eos_token_id=tokenizer.encode('<|im_end|>')[0],
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)