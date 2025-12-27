import torch 
import torch.nn as nn 
import sys 
from pathlib import Path

parent_folder = Path(__file__).resolve().parent
sys.path.insert(0, str(parent_folder))
print('parent_folder ', parent_folder)


from transformer_block.transformer_block import TransformerBlock
from layer_norm.layer_norm import LayerNorm



class GPT2(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(num_embeddings=cfg.vocab_size, embedding_dim=cfg.emb_dim)
        self.pos_emb = nn.Embedding(num_embeddings=cfg.context_length, embedding_dim=cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)

        # Transformer block only works in emb_dim space
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )

        # LayerNorm 
        self.final_norm = LayerNorm(cfg.emb_dim)

        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    
    def forward(self, in_idx): 
        batch_size, seq_len = in_idx.shape 

        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits 
     