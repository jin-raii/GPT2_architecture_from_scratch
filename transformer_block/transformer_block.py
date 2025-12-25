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
        self.attn = MultiHeadAttention(
            d_in=cfg.vocab_size, 
            d_out=cfg.emb_dim, 
            context_length=cfg.context_length, 
            dropout=cfg.drop_rate, 
            num_heads=cfg.num_heads, 
            qkv_bias=cfg.qkv_bias
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)


    def forward(self, x): 
        pass 