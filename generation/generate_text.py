import torch 


def generate_text(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array 
    for _ in range(max_new_tokens):
        # crop current context if it exceeds the suppported context size
        # E.g. if LLM supports only 5 tokens, and the context size is 10 
        # then only the last 5 tokens are used as context 
        idx_cond = idx[:, -context_size:]

        # get the predictions 
        with torch.no_grad():
            logits = model(idx_cond)

        
        # focus only on the last token 
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # get the idx of the vocab entry with the highest logits value 
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # append sampled index to the running sequence 
        idx = torch.cat((idx, idx_next), dim=1) # (b, n_tokens+1)

        return idx 