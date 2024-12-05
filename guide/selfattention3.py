#selfattention 的第三种写法（细节）
import torch.nn as nn
import torch.nn.functional as F
import math
import torch   
class selfattention3(nn.Module):
    def __init__(self,dim:int,dropout:float=0.1,max_seq=512)->None:
        super().__init__()
        self.dim = dim
        self.q = nn.Linear(dim,dim*3)
        self.max_seq = max_seq
        self.dropout = nn.Dropout(dropout)
        self.mask = torch.full((self.max_seq,self.max_seq),float("-inf"))
        self.mask = torch.triu(self.mask,diagonal=1)

    def forward(self,x,sqence_length):
        qkv=self.q(x)
        q,k,v = torch.chunk(qkv,3,dim=-1)
        attentionweight=q @ k.transpose(-1,-2) /math.sqrt(self.dim)

        
        attentionweight +=self.mask[:sqence_length,:sqence_length]
        print(attentionweight)
        attentionweight = torch.softmax(attentionweight,dim=-1)

        print(attentionweight)
        attentionweight = self.dropout(attentionweight)
        return attentionweight @ v


x= torch.randn(2,3,4)    
model = selfattention3(4)
print(model(x,x.shape[-2]))
