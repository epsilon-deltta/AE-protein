
def get_vocab_map(vocab_path='./data/vocab.txt'):
    with open(vocab_path,'r') as f:
        vocab = f.read()

    vocab = vocab.replace('\n','')

    import re
    p = re.compile('\s+')
    vocab = re.sub(p,' ',vocab)

    import ast
    vocab = ast.literal_eval(vocab.split('=')[1].strip())

    len_vocab = len(vocab)
    vocab_map = dict(zip(vocab,range(len_vocab)))
    return vocab_map
vocab_map =get_vocab_map()

import torch
from torch.nn import functional as F
import pandas as pd
class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self,path,transform='oneHot',only=None):
        self.df        = pd.read_csv(path)
        if type(transform) == str:
            self.transform = transform.lower()
        else:
            self.transform = transform
        
        if self.transform == 'ae':
            if only is None:
                only = 1
        
        if only is not None:
            self.df = self.df[self.df.label == only]
            
    def __getitem__(self,idx):
        item = self.df.iloc[idx]
        x = item['seq']
        y = item['label']
        
        if self.transform == 'onehot':
            x = self.seq2oneHot(x)
        elif self.transform is None:
            x = self.seq2int(x)
        elif self.transform == 'ae':
            x = self.seq2oneHot(x)
            x = x.to(torch.float)
            y = x 
            return x,y
        y = torch.tensor(y)
        # y = self.label2oneHot(y)
        return x,y
    
    def __len__(self):
        return len(self.df)
    
    def seq2int(self,seq):
        seq2int = [ vocab_map[x] for x in list(seq) ]
        seq2int = torch.tensor(seq2int)
        return seq2int
    
    def seq2oneHot(self,seq): # 'ABCDE..' -> OneHot (10,20) 
        seq2int = self.seq2int(seq)
        oneHot = F.one_hot(seq2int,num_classes=len(vocab_map) )
        return oneHot
    
    def label2oneHot(self,label):
        return F.one_hot(torch.tensor(label),num_classes=2)
    
   