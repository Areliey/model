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
from rotary import apply_rotary_emb

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'
        
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


def lambda_init(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * (depth - 1))


class MyQwen2DiffAttn(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super ().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = self.hidden_size // self.num_heads
        self.layer_idx = layer_idx if layer_idx is not None else 0
        self.lambda_init = lambda_init(self.layer_idx)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # 替换原始QKV投影层
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_size, bias=True, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_size, bias=True, dtype=torch.bfloat16)

        # 保持原始输出投影结构
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype=torch.bfloat16)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_size, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_size, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_size, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_size, dtype=torch.float32).normal_(mean=0, std=0.1))

        # 保持原始配置参数
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.attention_dropout)
        self.subln = RMSNorm(self.head_size, eps=1e-5, elementwise_affine=False)

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

        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.num_key_value_heads, self.head_size).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.num_key_value_heads, self.head_size).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(v, position_ids)
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        split_size = self.head_size // 2
        q1, q2 = q.split(split_size, dim=-1)
        k1, k2 = k.split(split_size, dim=-1)

        scale = 1.0 / math.sqrt(self.head_size)

        att1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        att2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale

        if attention_mask is not None:
            att1 = att1 + attention_mask
            att2 = att2 + attention_mask
        else:
            causal_mask = torch.tril(torch.ones(T, T, device=hidden_states.device)).view(1, 1, T, T)
            att1 = att1.masked_fill(causal_mask == 0, float('-inf'))
            att2 = att2.masked_fill(causal_mask == 0, float('-inf'))

        att1 = F.softmax(att1, dim=-1)
        att2 = F.softmax(att2, dim=-1)
        att1 = self.attention_dropout(att1)
        att2 = self.attention_dropout(att2)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        att = att1 - lambda_full * att2
        
        y = torch.matmul(att, v)  # [B, n_head, T, head_size]
        y = y * (1 - self.lambda_init)

        y = y.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        attn_output = self.o_proj(y)
        y = self.resid_dropout(attn_output)

        if not output_attentions:
            att = None

        return y, att

# class MyQwen2DiffAttn(nn.Module):
#     def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
#         super ().__init__()

#         self.hidden_size = config.hidden_size
#         self.num_heads = config.num_attention_heads
#         self.head_size = self.hidden_size // self.num_heads // 2
#         self.layer_idx = layer_idx if layer_idx is not None else 0
#         self.lambda_init = lambda_init(self.layer_idx)
#         self.num_key_value_heads = config.num_key_value_heads # num_kv_heads
#         self.num_key_value_groups = self.num_heads // self.num_key_value_heads # n_rep

#         # 替换原始QKV投影层
#         self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype=torch.bfloat16)
#         self.k_proj = nn.Linear(self.hidden_size, self.hidden_size // self.num_key_value_groups, bias=False, dtype=torch.bfloat16)
#         self.v_proj = nn.Linear(self.hidden_size, self.hidden_size // self.num_key_value_groups, bias=False, dtype=torch.bfloat16)

#         # 保持原始输出投影结构
#         self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype=torch.bfloat16)
#         self.lambda_q1 = nn.Parameter(torch.zeros(self.head_size, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k1 = nn.Parameter(torch.zeros(self.head_size, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_q2 = nn.Parameter(torch.zeros(self.head_size, dtype=torch.float32).normal_(mean=0, std=0.1))
#         self.lambda_k2 = nn.Parameter(torch.zeros(self.head_size, dtype=torch.float32).normal_(mean=0, std=0.1))

#         # 保持原始配置参数
#         self.attention_dropout = nn.Dropout(config.attention_dropout)
#         self.resid_dropout = nn.Dropout(config.attention_dropout)
#         self.subln = RMSNorm(2*self.head_size, eps=1e-5, elementwise_affine=False)

#     def forward(
#             self,
#             hidden_states: torch.Tensor,
#             attention_mask: Optional[torch.Tensor] = None,
#             position_ids: Optional[torch.LongTensor] = None,
#             past_key_value: Optional[Tuple[torch.Tensor]] = None,
#             output_attentions: bool = False,
#             cache_position: Optional[torch.LongTensor] = None,
#             position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
#             use_cache: bool = False,
#             **kwargs
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

#         B, T, embed_dim = hidden_states.shape

#         q = self.q_proj(hidden_states).view(B, T, 2 * self.num_heads, self.head_size)
#         k = self.k_proj(hidden_states).view(B, T, 2 * self.num_key_value_heads, self.head_size)
#         v = self.v_proj(hidden_states).view(B, T, self.num_key_value_heads, 2 * self.head_size)

#         # if position_embeddings is None:
#         #     cos, sin = self.rotary_emb(v, position_ids)
#         #     print("11111")
#         # else:
#         #     cos, sin = position_embeddings
        
#         # q = apply_rotary_emb(q, cos, sin, interleaved=True)
#         # k = apply_rotary_emb(k, cos, sin, interleaved=True)

#         # if past_key_value is not None:
#         #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  
#         #     k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

        
#         scale = 1.0 / math.sqrt(self.head_size)
#         q=q.transpose(1,2)
#         k = repeat_kv(k.transpose(1,2), self.num_key_value_groups)
#         v = repeat_kv(v.transpose(1,2), self.num_key_value_groups)
#         q *= scale
#         attn_weights = torch.matmul(q, k.transpose(-1, -2))

#         if attention_mask is None:
#             attention_mask = torch.triu(
#                 torch.zeros([T, T])
#                 .float()
#                 .fill_(float("-inf"))
#                 .type_as(attn_weights),
#                 1,
#             )

#         attn_weights = torch.nan_to_num(attn_weights)
#         attn_weights += attention_mask   
#         attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
#             attn_weights
#         )

#         lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
#         lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
#         lambda_full = lambda_1 - lambda_2 + self.lambda_init
#         attn_weights = attn_weights.view(B, self.num_heads, 2, T, T)
#         attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]

#         attn = torch.matmul(attn_weights, v)
#         attn = self.subln(attn)
#         attn = attn * (1 - self.lambda_init)
#         attn = attn.transpose(1, 2).reshape(B, T, self.num_heads * 2 * self.head_size)
#         attn_output = self.o_proj(attn)

#         return attn, attn_output
        


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
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_name)

    modified_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    index = 1
    for layer in modified_model.model.layers:
        layer.self_attn = MyQwen2DiffAttn(config, index).to(device)
        index = index + 1
    
    print(modified_model)
    modified_model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法

    ds = load_dataset("Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT")
    ds = ds['train'].select_columns(['instruction', 'input', 'output'])
    ds = ds.select(range(11000))
    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
    print(tokenized_id)


    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2"],
        inference_mode=False, # 训练模式
        r=8, # Lora 秩
        lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1# Dropout 比例
    )
    modified_model = get_peft_model(modified_model, config)
    modified_model.print_trainable_parameters()
    args = TrainingArguments(
        output_dir="./output/Qwen2.5",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=5000,
        learning_rate=1e-4,
        save_on_each_node=False,
        gradient_checkpointing=True
    )

    trainer = Trainer(
        model=modified_model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()

    peft_model_id = "./Qwen_lora"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
