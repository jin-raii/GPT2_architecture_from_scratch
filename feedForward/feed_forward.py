import torch.nn as nn 
import sys 
from pathlib import Path 

parent_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(parent_dir))

from activation.gelu_activation import GELU


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(
                in_features=cfg.emb_dim, 
                out_features=4 * cfg.emb_dim
            ), 
            GELU(), 
            nn.Linear(
                in_features=4 * cfg.emb_dim, 
                out_features=cfg.emb_dim 
            )
        )

    def forward(self, x): 
        return self.layers(x)
    
