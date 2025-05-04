import math
import os

import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from transformers import Qwen2Config, AutoModelForCausalLM, AutoModel, AutoConfig, Qwen2ForCausalLM, AutoTokenizer
from typing import Optional, Tuple
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from datasets import load_dataset
from safetensors.torch import load_model, save_model
from peft import PeftModel, PeftConfig

def process_func(example):
    MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# 使用示例
if __name__ == "__main__":
    device = "cuda"  # the device to load the model onto

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    original_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    print(original_model)
    original_model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT")
    ds = ds['train'].select_columns(['instruction', 'input', 'output'])
    ds = ds.select(range(11000))
    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
    print(tokenized_id)

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False, # 训练模式
        r=8, # Lora 秩
        lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1# Dropout 比例
    )
    original_model = get_peft_model(original_model, config)
    original_model.print_trainable_parameters()
    
    args = TrainingArguments(
        output_dir="./output/original_Qwen2.5",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=10000,
        learning_rate=1e-4,
        save_on_each_node=False,
        gradient_checkpointing=True
    )

    trainer = Trainer(
        model=original_model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()

    peft_model_id = "./original_Qwen_lora"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

