import torch
import torch.nn as nn
import torch.nn.functional as F

class ModifiedAttention(nn.Module):
    """Modified version that keeps checkpoint compatibility and supports cross-attention"""
    def __init__(self, embed_dims, num_heads, attn_drop=0., proj_drop=0., 
                 batch_first=True, sr_ratio=1):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=attn_drop, 
            batch_first=batch_first
        )
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio

        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                embed_dims, embed_dims, kernel_size=sr_ratio, 
                stride=sr_ratio
            )
            self.norm = nn.LayerNorm(embed_dims)
            
        self.conv_q = nn.Conv2d(
            embed_dims, embed_dims, 
            kernel_size=3, padding=1, groups=embed_dims
        )
        self.conv_k = nn.Conv2d(
            embed_dims, embed_dims, 
            kernel_size=3, padding=1, groups=embed_dims
        )
        self.conv_v = nn.Conv2d(
            embed_dims, embed_dims, 
            kernel_size=3, padding=1, groups=embed_dims
        )
        
        nn.init.kaiming_normal_(self.conv_q.weight)
        nn.init.kaiming_normal_(self.conv_k.weight)
        nn.init.kaiming_normal_(self.conv_v.weight)

    def forward(self, query_input, key_value_input, H, W):
        """
        query_input: tensor for queries (e.g., from decoder)
        key_value_input: tensor for keys and values (e.g., from encoder)
        H, W: height and width of the feature map for reshaping
        """
        B, N, C = query_input.shape
        
        q = query_input
        k = v = key_value_input
 
        q_2d = q.reshape(B, H, W, C).permute(0, 3, 1, 2)
        k_2d = k.reshape(B, H, W, C).permute(0, 3, 1, 2)
        v_2d = v.reshape(B, H, W, C).permute(0, 3, 1, 2)

        q_conv = self.conv_q(q_2d).permute(0, 2, 3, 1).reshape(B, N, C)
        k_conv = self.conv_k(k_2d).permute(0, 2, 3, 1).reshape(B, N, C)
        v_conv = self.conv_v(v_2d).permute(0, 2, 3, 1).reshape(B, N, C)
        
        if self.sr_ratio > 1:
            x_ = query_input.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            k = v = x_

        q = q + q_conv
        k = k + k_conv
        v = v + v_conv
        
 
        out = self.attn(q, k, v)[0]

        out = self.proj_drop(out)
        
        return out

def modify_backbone_attention(model):
    """Helper function to modify model while maintaining checkpoint compatibility"""
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            print(" ____________________________ ")
            parent_name = '.'.join(name.split('.')[:-1])
            parent = model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)
                    
            if hasattr(parent, 'attn'):

                embed_dims = module.embed_dim
                num_heads = module.num_heads
                sr_ratio = getattr(parent, 'sr_ratio', 1)
                

                new_attn = ModifiedAttention(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    sr_ratio=sr_ratio
                )

                new_attn.attn.load_state_dict(module.state_dict())
                

                parent.attn = new_attn
                
    return model