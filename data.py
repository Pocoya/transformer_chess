import torch
from torch.utils.data import Dataset, DataLoader
import os
import json

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3


CHESS_VOCAB = "<pad> <unk> <bos> <eos> 0 1 2 3 4 5 6 7 8 9 / a b c d e f g h p n b r q k P N B R Q K w - K Q q k  ".split(' ')
char_to_id = {char: i for i, char in enumerate(CHESS_VOCAB) if char != ''}
if ' ' not in char_to_id:
    char_to_id[' '] = len(char_to_id)
id_to_char = {i: char for char, i in char_to_id.items()}


class ChessTokenizer:
    """Simple character-level tokenizer for chess FEN and moves"""
    def __init__(self):
        self.vocab_size = len(CHESS_VOCAB)
    
    def encode(self, text, max_len):

        ids = [char_to_id.get(c, UNK_IDX) for c in text]
        ids = [BOS_IDX] + ids + [EOS_IDX]


        if len(ids) < max_len:
            ids += [PAD_IDX] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return ids
    

    def decode(self, ids):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        # Ignore special tokens 0-3
        return "".join([id_to_char.get(i, "") for i in ids if i > 3])


class ChessDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, num_samples=5000000, max_len_src=100, max_len_tgt=10):
        self.src_tensors = []
        self.tgt_tensors = []

        print(f"Loading and Tokenizing {num_samples} Chess positions...")
        with open(jsonl_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                data = json.loads(line)

                # src = FEN (board state)
                # tgt = UCI move (e2e4)
                s_ids = tokenizer.encode(data['src'], max_len_src)
                t_ids = tokenizer.encode(data['tgt'], max_len_tgt)

                self.src_tensors.append(torch.tensor(s_ids, dtype=torch.int16))
                self.tgt_tensors.append(torch.tensor(t_ids, dtype=torch.int16))

                if i % 500000 == 0 and i > 0:
                    print(f"Loaded {i} samples...")
    
    def __len__(self):
        return len(self.src_tensors)
    
    def __getitem__(self, idx):
        return{
            'src': self.src_tensors[idx].long(),
            'tgt_input': self.tgt_tensors[idx][:-1].long(),
            'tgt_output': self.tgt_tensors[idx][1:].long()
        }


def  get_dataloaders(batch_size=256, num_samples=5000000):
    """Prepares Chess DataLoaders"""
    jsonl_file = "chess_train_data.jsonl"
    if not os.path.exists(jsonl_file):
        raise FileNotFoundError(f"Run the extraction script first to create {jsonl_file}")

    tokenizer = ChessTokenizer()

    # Splitting the data, total samples = train + val
    full_ds = ChessDataset(jsonl_file, tokenizer, num_samples=num_samples)
    actual_total = len(full_ds)

    # Calculate split sizes and split
    val_len = int(actual_total * 0.01) 
    train_len = actual_total - val_len
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_len, val_len]) # Split

    print(f"Dataset size: Train: {len(train_ds)} | Val: {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, tokenizer

