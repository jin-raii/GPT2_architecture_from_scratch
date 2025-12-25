import torch.nn as nn 
import torch 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        assert (d_out % num_heads == 0), 'd_out must be divisible by num_heads'

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads 

        self.W_query = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.W_key = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.W_value = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)

        # combine head outputs 
        self.out_proj = nn.Linear(in_features=d_out, out_features=d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(name='mask', tensor=torch.triu(torch.ones(context_length, context_length), diagonal=1))


    def forward(self, x): 
        b, num_tokens, d_in = x.shape 
        
        keys = self.W_key(x) 
        queries = self.W_query(x)
        values = self.W_value(x)


        # split matrix by adding num_heads dimension 
        # unroll last dim: (b, num_tokens, d_out) --> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) # we are splitting d_out to head_dim e.g(d_out = 4 --> head_dim = 2 & head_dim = 2 )
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # transpose : (b_num_tokens, num_head, head_dim) --> (b, num_head, num_tokens, head_dim)
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        # compute scaled dot-product attn (aka self-attention) with causal mask 
        attn_score = queries @ keys.transpose(2,3) 

        # original mask truncated to the number of tokens and converted to boolean 
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # use the mask to fill attention scores 
        attn_score.masked_fill(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_score / keys.shape[-1]**0.5, dim=-1) # scalling and softmax 
        attn_weights = self.dropout(attn_weights)

        # shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1,2)

        # combine heads, where self.d_out = self.num_heads * self.head_dim 
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out) # re-arrange the memory with contiguous if use view else can replace view with reshape 
        context_vec = self.out_proj(context_vec)

        return context_vec
        





