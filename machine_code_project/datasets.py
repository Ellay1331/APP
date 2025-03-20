# -*- coding: utf-8 -*-
import os, json, re, logging, string, torch
from torch.utils.data import Dataset

# Используем те же константы, что и в model.py
CHAR_VOCAB = list(string.printable)
base_vocab_size = len(CHAR_VOCAB)
PAD_TOKEN, SOS_TOKEN, EOS_TOKEN = "<PAD>", "<SOS>", "<EOS>"
PAD_IDX, SOS_IDX, EOS_IDX = base_vocab_size, base_vocab_size + 1, base_vocab_size + 2

def remove_c_comments(code):
    pattern = r'//.*?$|/\*.*?\*/'
    return re.sub(pattern, '', code, flags=re.MULTILINE | re.DOTALL)

def remove_python_comments(code):
    return re.sub(r'#.*', '', code)

class JSONCCodeDataset(Dataset):
    def __init__(self, json_file, max_len=1024):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        code = self.data[idx].get("code", "")
        code = code[:self.max_len-2] if len(code) > self.max_len-2 else code
        seq = [SOS_IDX] + [ord(c) for c in code] + [EOS_IDX]
        return torch.tensor(seq, dtype=torch.long)

class RawCCodeDataset(Dataset):
    def __init__(self, data_dir, max_len=1024):
        self.max_len = max_len
        self.files = []
        allowed_ext = (".c", ".cpp", ".h", ".py", ".java", ".js", ".cs", ".rb", ".go", ".swift")
        for root, _, files in os.walk(data_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in allowed_ext:
                    self.files.append(os.path.join(root, f))
                    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        ext = os.path.splitext(path)[1].lower()
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logging.error(f"Ошибка чтения {path}: {e}")
            content = ""
        if ext in ('.c', '.cpp', '.h', '.java', '.js', '.cs', '.go', '.swift'):
            content = remove_c_comments(content)
        elif ext in ('.py', '.rb'):
            content = remove_python_comments(content)
        content = content[:self.max_len-2] if len(content) > self.max_len-2 else content
        seq = [SOS_IDX] + [ord(c) for c in content] + [EOS_IDX]
        return torch.tensor(seq, dtype=torch.long)

class RawTextDataset(Dataset):
    def __init__(self, data_dir, max_len=1024):
        self.max_len = max_len
        self.files = [os.path.join(root, f) for root, _, files in os.walk(data_dir)
                      for f in files if f.lower().endswith(".txt")]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            with open(self.files[idx], 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logging.error(f"Ошибка чтения {self.files[idx]}: {e}")
            content = ""
        content = content[:self.max_len-2] if len(content) > self.max_len-2 else content
        seq = [SOS_IDX] + [ord(c) for c in content] + [EOS_IDX]
        return torch.tensor(seq, dtype=torch.long)

def collate_fn_text(batch):
    lengths = [len(seq) for seq in batch]
    max_len = max(lengths)
    inputs, targets = [], []
    for seq in batch:
        pad = torch.full((max_len - len(seq),), PAD_IDX, dtype=torch.long)
        padded = torch.cat([seq, pad])
        inputs.append(padded)
        tgt = torch.cat([padded[1:], torch.tensor([PAD_IDX])])
        targets.append(tgt)
    return torch.stack(inputs), torch.stack(targets), torch.tensor(lengths, dtype=torch.long)
