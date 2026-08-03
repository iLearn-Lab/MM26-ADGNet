from thop import profile
from net import Net
import torch
import time

print('==> Building model..')
model = Net().cuda()
model = model.eval()
B = 1

dummy_input = torch.zeros(B, 1, 256, 256).cuda()

# Simulated CLIP text encoder output features
text_sequence = torch.randn(B, 20, 512).cuda()   # [B, seq_len, text_dim]
text_eot = torch.randn(B, 512).cuda()             # [B, text_dim]
l_mask_fg = torch.ones((B, 20), dtype=torch.long).cuda()
l_mask_bg = torch.ones((B, 20), dtype=torch.long).cuda()

# FPS benchmark
num_runs = 10
start_time = time.time()
for _ in range(num_runs):
    _ = model.forward(dummy_input, text_sequence, l_mask_fg, text_eot, l_mask_bg)
end_time = time.time()
total_time = end_time - start_time
fps = num_runs / total_time * B
print(f"FPS: {fps:.2f}")

# FLOPs and parameter count
flops, params = profile(model, (dummy_input, text_sequence, l_mask_fg, text_eot, l_mask_bg))
print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))
