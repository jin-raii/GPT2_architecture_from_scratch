import torch.nn as nn 
import sys 
from pathlib import Path

parent_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(parent_dir))

from attention.multihead_attn import MultiHeadAttention
from layer_norm.layer_norm import LayerNorm
from feedForward.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        print(f'vocab size : {cfg}')
        self.attn = MultiHeadAttention(
            d_in=cfg.vocab_size, 
            d_out=cfg.emb_dim, 
            context_length=cfg.context_length, 
            dropout=cfg.drop_rate, 
            num_heads=cfg.n_heads, 
            qkv_bias=cfg.qkv_bias
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_resid = nn.Dropout(cfg.drop_rate)

    def forward(self, x): 
        # shortcut connection for attention block 
        shortcut = x 
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_resid(x)
        x = x + shortcut # add the original input back 

        # shortcut connectin for feedforward block 
        shortcut = x 
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut

        return x 
    
