class GPT2Config():
    """
    config for the original GPT-2 'small' model 124M parameters.
    
    """

    emb_dim: int        = 768
    n_layers: int       = 12 
    n_heads: int        = 12 
    vocab_size: int     = 50257
    context_length: int = 1024
    drop_rate: int      = 0.1 
    qkv_bias: bool      = False 