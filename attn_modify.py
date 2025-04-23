import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from transformers import Qwen2Config, AutoModelForCausalLM, AutoModel, AutoConfig, Qwen2ForCausalLM, AutoTokenizer
from typing import Optional, Tuple
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from bigmodelvis import Visualization
from datasets import load_dataset
from safetensors.torch import load_model, save_model

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


class MyQwen2Attn(nn.Module):
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
        # 按 head_size 维度将 q 分割为 q1 和 q2
        split_size = self.head_size // 2
        q1, q2 = q.split(split_size, dim=-1)
        k1, k2 = k.split(split_size, dim=-1)
        # 缩放因子
        scale = 1.0 / math.sqrt(self.head_size)

        # 计算注意力分数
        att1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        att2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale

        # 因果掩码处理
        if attention_mask is not None:
            att1 = att1 + attention_mask
            att2 = att2 + attention_mask
        else:
            causal_mask = torch.tril(torch.ones(T, T, device=hidden_states.device)).view(1, 1, T, T)
            att1 = att1.masked_fill(causal_mask == 0, float('-inf'))
            att2 = att2.masked_fill(causal_mask == 0, float('-inf'))

        # Softmax归一化
        att1 = F.softmax(att1, dim=-1)
        att2 = F.softmax(att2, dim=-1)
        att1 = self.attention_dropout(att1)
        att2 = self.attention_dropout(att2)

        # lambda_full = self.lambda_init
        #
        # lambda_q1, lambda_q2 = self.q_proj_no_perm.split(self.head_size, dim=-1)
        # lambda_k1, lambda_k2 = self.k_proj_no_perm.split(self.head_size, dim=-1)

        # λ参数计算
        # lambda_1 = torch.exp(torch.sum(lambda_q1 * lambda_k1, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        # lambda_2 = torch.exp(torch.sum(lambda_q2 * lambda_k2, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_full = self.lambda_init
        # 组合注意力
        att = att1 - lambda_full * att2

        # 值向量加权
        y = torch.matmul(att, v)  # [B, n_head, T, head_size]
        y = self.subln(y)
        y = y * (1 - self.lambda_init)

        # 输出投影
        y = y.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        attn_output = self.o_proj(y)
        y = self.resid_dropout(attn_output)

        if not output_attentions:
            att = None
        #
        # print(self.layer_idx)
        # print("att shape:", y.shape)  # 应为 [B, num_heads, T, T]
        # print("v shape:", v.shape)  # 应为 [B, num_heads, T, head_size]

        # 保持与原始输出格式一致
        return y, None, past_key_value


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)


# 使用示例
if __name__ == "__main__":
    # # 替换类定义
    device = "cuda"  # the device to load the model onto

    model_name = "Qwen/Qwen2-0.5B-Instruct"
    # # modeling_qwen2.QWEN2_ATTENTION_CLASSES["sdpa"] = MyQwen2Attn
    # # modeling_qwen2.Qwen2SdpaAttention = MyQwen2Attn
    #
    os.environ["http_proxy"] = "http://127.0.0.1:7897"
    os.environ["https_proxy"] = "http://127.0.0.1:7897"

    original_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    # # model.save_pretrained("./model")
    model_vis = Visualization(original_model)
    model_vis.structure_graph()

    modified_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    index = 1
    for layer in modified_model.model.layers:
        layer.self_attn = MyQwen2Attn(config, index).to(device)
        index = index + 1

    model_vis = Visualization(modified_model)
    model_vis.structure_graph()
    modified_model.to(device)
    # # 前向传播测试
    # input_ids = torch.randint(0, 1000, (1, 64)).to(device)
    # output = modified_model(input_ids)
    # print(f"输出形状：{output.logits.shape}")


    prompt = "给我介绍一下什么是大预言模型？"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids =modified_model.generate(
        model_inputs.input_ids,
        max_new_tokens=60
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
