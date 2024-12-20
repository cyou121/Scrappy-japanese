import torch
import torch.nn as nn
import torch
import math


class MutiHeadattention(nn.Module):
    def __init__(self,hidden_dim: int=12,head_num: int=3,max_seq_len:int=4)->None:
        super().__init__()
        self.hidden_dim=hidden_dim
        self.head_num=head_num
        self.head_dim=hidden_dim//head_num

        self.q_proj=nn.Linear(self.head_dim,hidden_dim)
        self.k_proj=nn.Linear(self.head_dim,hidden_dim)
        self.v_proj=nn.Linear(self.head_dim,hidden_dim)
        
        self.att_dropout=nn.Dropout(0.1)
        self.out_proj=nn.Linear(hidden_dim,self.head_dim)
        
        mask = torch.full((max_seq_len, max_seq_len), float("-inf"))  
        mask = torch.triu(mask, diagonal=1)  
        self.k_cache=None
        self.v_cache=None
        self.use_kvcache=None
        self.register_buffer("mask", mask)  

    def forward(self,x):
        batch_size,seq_len,_ =x.shape

        if self.use_kvcache and self.eval():
            if self.k_cache and self.v_cache is not None:
                token=x[:,-1,:]
                q=torch.cat(torch.zeros_like(x[:, :-1, :]),self.q_proj(token),dim=1)
                k=torch.cat(self.k_cache,self.k_proj(token),dim=1)
                v=torch.cat(self.v_cache,self.v_proj(token),dim=1)
            else:
                q=self.q_proj(x)
                k=self.k_proj(x)
                v=self.v_proj(x)
            self.k_cache=k
            self.v_cache=v
        else:
            q=self.q_proj(x)
            k=self.k_proj(x)
            v=self.v_proj(x)


        print("1")
        # q: [batch_size,seq_len,hidden_dim]
        # k: [batch_size,seq_len,hidden_dim]
        # v: [batch_size,seq_len,hidden_dim]
        
        q=q.view(batch_size,seq_len,self.head_num,self.head_dim).transpose(1,2)
        k=k.view(batch_size,seq_len,self.head_num,self.head_dim).transpose(1,2)
        v=v.view(batch_size,seq_len,self.head_num,self.head_dim).transpose(1,2)
        print("2")

        # q: [batch_size,elf.head_num,seq_len,s,self.head_dim]
        # k: [batch_size,elf.head_num,seq_len,s,self.head_dim]
        # v: [batch_size,elf.head_num,seq_len,s,self.head_dim]

        atten_weight=q@k.transpose(-2,-1)/math.sqrt(self.head_dim)
        atten_weight+=self.mask[:seq_len,:seq_len]
        
        atten_weight=torch.softmax(atten_weight,dim=-1)
        print(atten_weight)
        atten_weight=self.att_dropout(atten_weight)
        attention_output=atten_weight@v

        attention_output=attention_output.transpose(1,2).contiguous()

        attention_output=attention_output.view(batch_size,seq_len,self.hidden_dim)
        return self.out_proj(attention_output)


x=torch.randn(2,3,4)
print(x)
print("")
model=MutiHeadattention()
print(model(x))


###################Tips:
'''
x1 = torch.randint(low=0, high=3, size=(2,3,4,5)) 
x2 = torch.randint(low=0, high=3, size=(2,3,4,5)) 

For two tensors of the same shape, the batches and channels match up one-to-one. This means:

The first batch of x1 pairs with the first batch of x2.
The second batch of x1 pairs with the second batch of x2.
And similarly for channels:

The first channel of x1 pairs with the first channel of x2.
The second channel of x1 pairs with the second channel of x2.
The third channel of x1 pairs with the third channel of x2.
Without any special operations, there's no mixing of batches or channels across different indices.
'''
#example
import torch
x1 = torch.randint(low=0, high=3, size=(2,3,4,5)) 
x2 = torch.randint(low=0, high=3, size=(2,3,4,5))
print(x1.T.shape)
print((x1@x2.transpose(-1,-2)).shape)

print("")
x1 = torch.randint(low=0, high=3, size=(1,1,4,5)) 
x2 = torch.randint(low=0, high=3, size=(2,3,4,5))
print((x1@x2.transpose(-1,-2)).shape)

print("")
x1 = torch.randint(low=0, high=3, size=(1,1,4,5)) 
x2 = torch.randint(low=0, high=3, size=(2,3,4,5))
print((x1@x2.transpose(-1,-2)).shape)


x1=torch.randn(1,2,3)
print(x1)
x2=torch.randn(1,2,3)
print(x2)
token=torch.cat((x1,x2),dim=1)
print(token)
