#selfattention 的第二种写法（高效）
import torch.nn as nn
import torch.nn.functional as F
import math
import torch   
class selfattention2(nn.Module):
    def __init__(self,dim:int=4)->None:
        super().__init__()
        self.dim = dim
        self.q = nn.Linear(dim,dim*3)
    def forward(self,x):
        qkv=self.q(x)
        q,k,v = torch.chunk(qkv,3,dim=-1)
        # q,k,v=torch.split(qkv,self.dim,dim=-1)
        attentionweight=torch.softmax(q @ k.transpose(-1,-2) /math.sqrt(self.dim),dim=-1)
        print(attentionweight)
        return attentionweight @ v
       
x = torch.randn(2,3,4)
model = selfattention2()
print(model(x))

###################Tips
qkv = torch.randn(1,2,3*2)
print(qkv)
q,k,v = torch.chunk(qkv,3,dim=-1)
print("")
print(q,k,v)
