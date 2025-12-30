from torch.utils.data import Dataset
import torch



class InstructDataset(Dataset):
    def __init__(self, data, tokenizer):
        

        self.data = data 
        self.tokenizer = tokenizer
        self.encoded_texts = []
        for entry in data: 
            # print(entry)
            instruction_plus_input = self.format_data(entry)
            response_text = f"\n\n Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            # tokens = tokenizer.encode(full_text)
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    
    
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, index):
        return self.encoded_texts[index]
    

def collate_fn(batch, pad_token=50256,ignore_index=-100, allowed_mask_length=None, device='cpu'):
    # find the longest sequence on the batch 
    batch_max_length = max(len(item) + 1 for item in batch)

    #pad and prepare input 
    inputs_list, targets_lst = [], []

    for item in batch: 
        new_item = item.copy()
        new_item += [pad_token]

        padded = (
            new_item + [pad_token] * (batch_max_length - len(new_item))
        )

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # replace all but the first padding tokens in targets by ignore_index 
        mask = targets == pad_token
        indicies = torch.nonzero(mask).squeeze()
        if indicies.numel() > 1: 
            targets[indicies[1:]] = ignore_index

        # optionally truncate to maximum sequence length 
        if allowed_mask_length is not None: 
            inputs = inputs[:allowed_mask_length]
            targets = targets[:allowed_mask_length]


        inputs_list.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_list).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor