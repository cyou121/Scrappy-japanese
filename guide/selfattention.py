#selfattention 的第一种写法（最普通的）
import torch.nn as nn
import torch.nn.functional as F
import math
import torch    


class attention(nn.Module):
    def __init__(self,hidden_dim:int = 512)->None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query_proj= nn.Linear(hidden_dim,hidden_dim)
        self.key_proj = nn.Linear(hidden_dim,hidden_dim)
        self.value_proj = nn.Linear(hidden_dim,hidden_dim)
        
    
    def forward(self,x):
        #X shape is (batch_size,seq_len,hidden_dim)
        Q= self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        #K.transpose(-1,-2)  == transpose(-2,-1)
        # K @ Q == torch.matmul
        attention_value = Q @ K.transpose(-2,-1)
        #除以的作用是什么
        attention_weight = attention_value / math.sqrt(self.hidden_dim)
        #torch.softmax == F.softmax
        result = F.softmax(attention_weight,dim=-1)
        return result @ V


x=torch.randn(2,3,512)
atten = attention()
print(atten(x))


##########################Tips：
import torch
K = torch.randn(1,1,3)
print(K)
print("-----")
print(K.transpose(-1,-2))
print(K.transpose(-2,-1))
##########################Tips：
import torch
K = torch.randn(2,3)
Q = torch.randn(3,2)
print(K @ Q)
print("-----")
print(torch.matmul(K, Q))
##########################Tips：
import torch
import torch.nn.functional as F
K = torch.randn(2,3)
print(K)
print(F.softmax(K,dim=-1)) #列维度 相加为1
print(F.softmax(K,dim=0))  #行维度
print(torch.softmax(K,dim=-1))
print(torch.softmax(K,dim=-1))

