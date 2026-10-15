import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from einops import rearrange, repeat

SYSTEM_PROMPT = "You are a professional cardiologist. Analyze the 12-lead ECG and provide a clinical report."
USER_PROMPT = "Here is the ECG signal data. Please generate the diagnosis."


# ==========================================
# 1. 1D SE-ResNet18 Encoder
# ==========================================
class SEBlock1D(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.fc(x.mean(dim=-1)).view(b, c, 1)
        return x * y


class ResBlock1D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_c, out_c, 3, stride, 1, bias=False),
            nn.BatchNorm1d(out_c), nn.ReLU(inplace=True),
            nn.Conv1d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_c), SEBlock1D(out_c)
        )
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_c, out_c, 1, stride, bias=False),
            nn.BatchNorm1d(out_c)
        ) if stride != 1 or in_c != out_c else nn.Identity()

    def forward(self, x):
        return F.relu(self.net(x) + self.shortcut(x))


class ECGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 1000点 -> 500 -> 250 -> 125 -> 63
        self.model = nn.Sequential(
            nn.Conv1d(12, 64, 7, 2, 3, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            ResBlock1D(64, 64, 1), ResBlock1D(64, 128, 2),
            ResBlock1D(128, 256, 2), ResBlock1D(256, 512, 2)
        )

    def forward(self, x):
        return self.model(x)  # (B, 512, 63)


# ==========================================
# 2. 感知重采样器 (Perceiver Resampler)
# ==========================================
class LegacyPerceiverResampler(nn.Module):
    def __init__(self, dim, num_latents=8, num_heads=8):
        super().__init__()
        # 可学习的 Latent Queries (论文核心)
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        # 使用 Cross-Attention 让 Latents 去看 ECG 的特征图
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.norm_mlp = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, dim, L) -> (B, L, dim)
        x = rearrange(x, 'b d l -> b l d')
        b = x.shape[0]
        # 复制 Latents 到 batch 维度
        queries = repeat(self.latents, 'n d -> b n d', b=b)

        # Cross-Attention: Query=Latents, Key/Value=ECG Features
        attn_out, _ = self.attn(queries, x, x)
        x = self.norm(queries + attn_out)
        x = self.norm_mlp(x + self.mlp(x))
        return x  # (B, num_latents, dim)


# ==========================================
# 3. 完整的多模态微调模型
# ==========================================
class PerceiverResamplerBlock(nn.Module):
    """Cross-attend ECG context, then let latent tokens exchange information."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.latent_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )
        # Small initial gates stabilize newly inserted residual paths.
        self.cross_gate = nn.Parameter(torch.tensor(-2.0))
        self.self_gate = nn.Parameter(torch.tensor(-2.0))
        self.ffn_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, latents, context, key_padding_mask=None):
        query = self.latent_norm(latents)
        key_value = self.context_norm(context)
        update, _ = self.cross_attn(
            query, key_value, key_value, key_padding_mask=key_padding_mask, need_weights=False
        )
        latents = latents + torch.sigmoid(self.cross_gate) * update
        query = self.self_norm(latents)
        update, _ = self.self_attn(query, query, query, need_weights=False)
        latents = latents + torch.sigmoid(self.self_gate) * update
        return latents + torch.sigmoid(self.ffn_gate) * self.ffn(self.ffn_norm(latents))


class PerceiverResampler(nn.Module):
    """Position-aware multi-scale gated Perceiver for ECG feature compression."""
    def __init__(self, dim, num_latents=8, num_heads=8, depth=2, max_positions=128, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.empty(1, num_latents, dim))
        self.position_embedding = nn.Parameter(torch.empty(1, max_positions, dim))
        # Two receptive fields retain short QRS-like changes and longer rhythm context.
        self.local_short = nn.Conv1d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.local_long = nn.Conv1d(dim, dim, 7, padding=3, groups=dim, bias=False)
        self.context_norm = nn.LayerNorm(dim)
        self.context_gate = nn.Parameter(torch.tensor(-1.0))
        self.blocks = nn.ModuleList(
            PerceiverResamplerBlock(dim, num_heads, dropout=dropout) for _ in range(depth)
        )
        self.output_norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.latents, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x, padding_mask=None):
        """x is (B, dim, L); padding_mask is (B, L), with True marking padding."""
        if x.ndim != 3 or x.shape[1] != self.dim:
            raise ValueError(f"Expected x with shape (B, {self.dim}, L), got {tuple(x.shape)}")
        batch_size, _, length = x.shape
        context = rearrange(x, 'b d l -> b l d')
        positions = F.interpolate(
            self.position_embedding.transpose(1, 2), size=length,
            mode="linear", align_corners=False
        ).transpose(1, 2).to(dtype=x.dtype)
        local_features = self.local_short(x) + self.local_long(x)
        local_features = rearrange(local_features, 'b d l -> b l d')
        context = self.context_norm(
            context + positions + torch.sigmoid(self.context_gate) * local_features
        )
        latents = self.latents.expand(batch_size, -1, -1)
        for block in self.blocks:
            latents = block(latents, context, key_padding_mask=padding_mask)
        return self.output_norm(latents)  # (B, num_latents, dim)


class ECGReportSystem(nn.Module):
    def __init__(self, model_id="Qwen/Qwen3-1.7B"):
        super().__init__()
        # 1. ECG Encoder (fp32 initially)
        self.encoder = ECGEncoder()
        # 2. Perceiver Resampler
        self.resampler = PerceiverResampler(dim=512, num_latents=32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # 3. LLM (Load in bf16)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cuda:0"
        )
        self.proj = nn.Linear(512, self.llm.config.hidden_size)

        # 4. LoRA 微调配置
        peft_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05, task_type="CAUSAL_LM"
        )
        self.llm = get_peft_model(self.llm, peft_config)

        # 【核心】强制转换所有自定义模块为 bfloat16，确保精度对齐
        self.encoder.to(torch.bfloat16)
        self.resampler.to(torch.bfloat16)
        self.proj.to(torch.bfloat16)

    def get_ecg_embeddings(self, ecg):
        # 防御性转换数据类型
        ecg = ecg.to(torch.bfloat16)
        # 1. 提取 1D 特征
        features = self.encoder(ecg)  # (B, 512, L)
        # 2. 感知重采样压缩至 32 个 Token
        latents = self.resampler(features)  # (B, 32, 512)
        # 3. 映射到 LLM 维度
        return self.proj(latents)  # (B, 32, 1536)

    def forward(self, ecg, input_ids, labels):
        # 获取心电特征 Embedding
        ecg_embeds = self.get_ecg_embeddings(ecg)
        # 获取文本 Embedding (input_ids 必须是 Long)
        text_embeds = self.llm.get_input_embeddings()(input_ids.long())

        # 拼接特征: [ECG(32 Tokens), Text(Sequence)]
        combined_embeds = torch.cat([ecg_embeds, text_embeds], dim=1)

        # 构造 labels, ECG 部分屏蔽 Loss (-100)
        ecg_labels = torch.full(
            (ecg.shape[0], self.resampler.num_latents),
            -100, device=ecg.device, dtype=torch.long
        )
        combined_labels = torch.cat([ecg_labels, labels.long()], dim=1)

        # 前向传播计算 Loss
        outputs = self.llm(inputs_embeds=combined_embeds, labels=combined_labels)
        return outputs.loss

    @torch.no_grad()
    def generate_report(self, ecg_signals, max_new_tokens=150):
        self.eval()
        batch_size = ecg_signals.shape[0]  # 获取当前输入的实际 batch_size (比如 2)
        device = ecg_signals.device

        # 1. 提取 ECG Embedding -> 形状 [B, 32, 1536]
        ecg_embeds = self.get_ecg_embeddings(ecg_signals)

        # 2. 构造推理用的 Prompt
        inference_prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{USER_PROMPT}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # 初始生成的 text_ids 形状是 [1, SeqLen]
        text_ids = self.tokenizer(inference_prompt, return_tensors="pt").input_ids.to(device)

        # 3. 获取文本 Embedding -> 形状 [1, SeqLen, 1536]
        text_embeds = self.llm.get_input_embeddings()(text_ids)

        # 【核心修复】：将 text_embeds 从 1 扩展到与 ecg_embeds 相同的 batch_size
        # 使用 .expand 可以不占用额外内存地进行广播
        text_embeds = text_embeds.expand(batch_size, -1, -1)  # 变为 [B, SeqLen, 1536]

        # 4. 现在拼接就不会报错了
        # [B, 32, 1536] + [B, SeqLen, 1536] -> [B, 32 + SeqLen, 1536]
        inputs_embeds = torch.cat([ecg_embeds, text_embeds], dim=1)

        # 5. 生成结果
        output_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False
        )

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)


# ==========================================
# 4. 训练与测试演示
# ==========================================
if __name__ == "__main__":
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
    model = ECGReportSystem().to(device)

    # 准备假数据 (BatchSize=2, Leads=12, Points=1000)
    dummy_ecg = torch.randn(2, 12, 1000).to(device)
    dummy_text = ["The ECG shows sinus rhythm with no ST changes.", "Atrial fibrillation is detected."]

    tokens = tokenizer(dummy_text, return_tensors="pt", padding=True).to(device)

    # --- 训练测试 ---
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss = model(dummy_ecg, tokens.input_ids, tokens.input_ids)
    loss.backward()
    optimizer.step()
    print(f"微调训练 Loss: {loss.item():.4f}")

    # --- 推理测试 (生成报告) ---
    print("\n>>> 模型正在生成报告...")
    gen_reports = model.generate_report(dummy_ecg, tokenizer)
    for i, report in enumerate(gen_reports):
        print(f"生成的报告 {i + 1}: {report}")
