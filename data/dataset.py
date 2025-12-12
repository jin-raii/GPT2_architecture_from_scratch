from torch.utils.data import Dataset




class InstructDataset(Dataset):
    def __init__(self, data, tokenizer):
        super().__init__()

        self.data = data 

        self.encoded_texts = []
        for entry in data: 
            # print(entry)
            instruction_plus_input = self.format_data(entry)
            response_text = f"\n\n Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def format_data(self, data):
        # Alpeca style 
        instruction_text = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately complets the request."
            f"\n\n### Instruction:\n{data['instruction']}"
        )

        input_text = f"\n\n### Input:\n{data['input']}" if data['input'] else '' 
        combined_text = instruction_text + input_text
        return combined_text
    
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, index):
        return self.encoded_texts[index]