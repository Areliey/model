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

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    return hidden_states.repeat_interleave(repeats=n_rep, dim=1)  # 沿 head 维度扩展，保持 T 不变
    
class MyQwen2Attn(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super ().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = self.hidden_size // self.num_heads
        self.layer_idx = layer_idx if layer_idx is not None else 0
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # 替换原始QKV投影层
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_size, bias=True, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_size, bias=True, dtype=torch.bfloat16)

        # 保持原始输出投影结构
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype=torch.bfloat16)

        # 保持原始配置参数
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.attention_dropout)
        self.subln = nn.LayerNorm(self.head_size, elementwise_affine=False)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
            **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        B, T, _ = hidden_states.shape

        # 投影到QKV空间
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.num_key_value_heads, self.head_size).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.num_key_value_heads, self.head_size).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(v, position_ids)
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)
        # repeat k/v heads if n_kv_heads < n_heads
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        attn_weights = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.head_size)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : k.shape[-2]]
            attn_weights = attn_weights + causal_mask

        att = F.softmax(attn_weights, dim=-1)
        att = self.attention_dropout(att)

        attn_output = torch.matmul(att, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            att = None

        # 保持与原始输出格式一致
        return attn_output, att


# 使用示例
if __name__ == "__main__":
    device = "cuda"  # the device to load the model onto

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    original_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    config = AutoConfig.from_pretrained(model_name)
    index = 1
    for layer in original_model.model.layers:
        layer.self_attn = MyQwen2Attn(config, index).to(device)
        index = index + 1
        
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
        output_dir="./output/original_Qwen2.5_diy",
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

    peft_model_id = "./original_Qwen_lora_diy"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

