import torch
import torch.nn as nn
import torch.nn.functional as F
from .BS_Branch import *
from .TL_Branch import *

class AFA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim//2, dim, 1)

    def forward(self, x, global_feat, local_feat):   

        attn1 = self.conv1(global_feat)
        attn2 = self.conv2(local_feat)
        
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1)
        attn = self.conv(attn).sigmoid()
        return x * (1 + attn)

class DualStreamFusion(nn.Module):
    def __init__(self, 
                 visual_dim=128, 
                 text_dim=512, 
                 fusion_dim=128,  
                 dropout_global=0.1,
                 dropout_local=0.0):
        super(DualStreamFusion, self).__init__()
        

        self.global_branch = BS_Branch(
            visual_dim=visual_dim,
            text_dim=text_dim,
            proj_dim=fusion_dim
        )


        self.local_branch = TL_Branch(
            dim=visual_dim,
            v_in_channels=visual_dim,
            l_in_channels=text_dim,
            key_channels=fusion_dim,
            value_channels=fusion_dim,
            num_heads=1,
            dropout=dropout_local
        )

        self.final_proj = nn.Sequential(
            nn.Conv2d(fusion_dim * 2, visual_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(visual_dim),
            nn.ReLU(inplace=True)
        )

        self.fusion = AFA(fusion_dim)

    def forward(self, x, l_seq, l_vec, l_mask_seq, l_mask_vec):
        """
        Args:
            x: visual feature map [B, visual_dim, H, W]
            l_seq: CLIP token sequence features [B, N_words, text_dim]
            l_vec: CLIP global [EOT] feature [B, text_dim]
            l_mask_seq: token-level attention mask [B, N_words]
            l_mask_vec: global mask [B, 1]
        """
        B, C, H, W = x.shape
        x_yuan = x
        out_global = self.global_branch(x, l_vec)

        x_flat = x.flatten(2).permute(0, 2, 1)
        l_seq_perm = l_seq.permute(0, 2, 1)

        if l_mask_seq.dim() == 2:
            l_mask_seq = l_mask_seq.unsqueeze(-1)
        if l_mask_vec.dim() == 2:
            l_mask_vec = l_mask_vec.unsqueeze(-1)

        out_local_flat = self.local_branch(x_flat, l_seq_perm, l_mask_seq)
        
        out_local = out_local_flat.permute(0, 2, 1).view(B, -1, H, W)
        out = self.fusion(x_yuan, out_global, out_local)
        
        return out