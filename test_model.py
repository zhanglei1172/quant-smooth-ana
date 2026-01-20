"""Simple test to verify model can run"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/dataset/workspace/zhangl98/models/Qwen3-0.6B/"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype="auto", device_map="auto"
)

print(f"Model config: max_position_embeddings={model.config.max_position_embeddings}")
print(f"Model vocab size: {model.config.vocab_size}")

# 设置pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 创建测试输入
seq_len = 16
tokens = torch.randint(0, tokenizer.vocab_size, (1, seq_len))
print(f"Input shape: {tokens.shape}")
print(f"Input device: {tokens.device}")

# 获取embedding层的设备
embeddings = model.model.embed_tokens
embed_device = next(embeddings.parameters()).device
print(f"Embedding device: {embed_device}")

# 将输入移动到embedding层的设备
tokens = tokens.to(embed_device)
print(f"Input device after move: {tokens.device}")

# 尝试前向传播
try:
    with torch.no_grad():
        output = model(tokens)
    print(f"Output shape: {output.logits.shape}")
    print("Test passed!")
except Exception as e:
    print(f"Test failed: {e}")
    import traceback

    traceback.print_exc()
