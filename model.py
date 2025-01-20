import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass
from LMConfig import LMconfig
from typing import Any, Optional, Tuple


def precompute_pos_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) 
    t = torch.arange(end, device=freqs.device)  
    freqs = torch.outer(t, freqs).float()  
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  
    return pos_cis

def apply_rotary_emb(xq, xk, pos_cis):
    def unite_shape(pos_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) 
    pos_cis = unite_shape(pos_cis, xq_)  
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3) 
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)  
    return xq_out.type_as(xq), xk_out.type_as(xk)  




class RMS(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps  
        self.weight = nn.Parameter(torch.ones(dim))  

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  

    def forward(self, x):
        output = self._norm(x.float()).type_as(x) 
        return output * self.weight  
    
class MutiHeadattention(nn.Module):
    def __init__(self)->None:
        super().__init__()
        params=LMconfig()
        self.hidden_dim=params.dim
        self.falsh=params.if_falsh
        self.head_num=params.head_num
        self.head_dim=self.hidden_dim//self.head_num
        self.max_seq_len=params.max_seq_len

        self.q_proj=nn.Linear(self.hidden_dim,self.hidden_dim)
        self.k_proj=nn.Linear(self.hidden_dim,self.hidden_dim)
        self.v_proj=nn.Linear(self.hidden_dim,self.hidden_dim)
        self.dropout = params.dropout
        self.att_dropout=nn.Dropout(0.1)
        self.out_proj=nn.Linear(self.hidden_dim,self.hidden_dim)
        
        mask = torch.full((self.max_seq_len, self.max_seq_len), float("-inf"))  
        mask = torch.triu(mask, diagonal=1)  
        self.k_cache=None
        self.v_cache=None
        self.use_kvcache=None
        self.register_buffer("mask", mask)  

    def forward(self,x:torch.Tensor,pos_cis:torch.Tensor):
        batch_size,seq_len,_ =x.shape

        if self.use_kvcache and self.eval():
            if self.k_cache and self.v_cache is not None:
                token=x[:,-1,:]
                q=torch.cat((torch.zeros_like(x[:, :-1, :]),self.q_proj(token)),dim=1)
                k=torch.cat((self.k_cache,self.k_proj(token)),dim=1)
                v=torch.cat((self.v_cache,self.v_proj(token)),dim=1)
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

        #print("1")
        # q: [batch_size,seq_len,hidden_dim]
        # k: [batch_size,seq_len,hidden_dim]
        # v: [batch_size,seq_len,hidden_dim]
        
        q=q.view(batch_size,seq_len,self.head_num,self.head_dim)
        k=k.view(batch_size,seq_len,self.head_num,self.head_dim)
        v=v.view(batch_size,seq_len,self.head_num,self.head_dim)
        #print("2")
        # print(q.shape)
        q, k = apply_rotary_emb(q, k, pos_cis)  # 应用旋转位置编码

        q = q.transpose(1, 2)  # 调整 Q 的形状
        k = k.transpose(1, 2)  # 调整 K 的形状
        v = v.transpose(1, 2)  # 调整 V 的形状
        # print(q.shape)


        # q: [batch_size,elf.head_num,seq_len,s,self.head_dim]
        # k: [batch_size,elf.head_num,seq_len,s,self.head_dim]
        # v: [batch_size,elf.head_num,seq_len,s,self.head_dim]
        if self.falsh:
            attention_output=torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                                      dropout_p=self.dropout if self.training else 0.0,
                                                                      is_causal=True)
        else:
            atten_weight=q@k.transpose(-2,-1)/math.sqrt(self.head_dim)
            atten_weight+=self.mask[:seq_len,:seq_len]
            
            atten_weight=torch.softmax(atten_weight,dim=-1)
            atten_weight=self.att_dropout(atten_weight)
            attention_output=atten_weight@v





        attention_output=attention_output.transpose(1,2).contiguous()

        attention_output=attention_output.view(batch_size,seq_len,self.hidden_dim)
        return self.out_proj(attention_output)



class FFN(nn.Module):
    def __init__(self)-> None:
     super().__init__()
     params=LMconfig()
     self.Linear1=nn.Linear(params.dim,params.dim,bias=False)
     self.Linear2=nn.Linear(params.dim,params.dim,bias=False)
     self.Linear3=nn.Linear(params.dim,params.dim,bias=False)

     self.RMS=RMS(params.dim,eps=params.eps)
     self.dropout=nn.Dropout(0.1)


    def forward(self, x):
        start=x
        x=self.RMS(x)
        x1 = self.Linear1(x)
        x2 = self.Linear2(x)
        
        x2=F.silu(x2)
        x3=x1*x2
        x3=self.Linear3(x3)
        x3=self.dropout(x3)
        x3+=start

        return x3


class TransformerBlock(nn.Module):
    def __init__(self)-> None:
     super().__init__()
     params=LMconfig()
     self.MutiHeadattention=MutiHeadattention()
     self.FFN=FFN()
     self.attention_norm=RMS(params.dim,eps=params.eps)
     self.FFN_norm=RMS(params.dim,eps=params.eps)


    def forward(self, x:torch.Tensor,pos_cis)->torch.Tensor:
       x1= self.MutiHeadattention(self.attention_norm(x),pos_cis)+x
       x2= self.FFN(self.FFN_norm(x1))+x1
       return x2
        

        



class Transformer(nn.Module):
    def __init__(self)-> None:
        super().__init__()
        params=LMconfig()
        self.decoder_layers=nn.ModuleList([TransformerBlock() for i in range(params.n_layers)])
        self.rmsnorm=RMS(params.dim,params.eps)##1
        self.dropout=nn.Dropout(params.dropout)
        self.embedding=nn.Embedding(params.vocab_size,params.dim)
        
        pos_cis = precompute_pos_cis(params.dim // params.head_num, params.max_seq_len)
        self.register_buffer("pos_cis", pos_cis, persistent=False)  


        self.Linear=nn.Linear(params.dim,params.vocab_size,bias=False)
        self.OUT = CausalLMOutputWithPast()  

    def forward(self, token: Optional[torch.Tensor] = None, target: Optional[torch.Tensor] = None,**keyargs) ->torch.Tensor:
  
        try:
         _bszize, seqlen = token.shape
        except ValueError as e:
          print("训练时，输入的Size有错,训练中断")
        
            
        token=self.embedding(token)

        token=self.dropout(token)
        
        pos_cis = self.pos_cis[:seqlen]


        for _,decoder_layers in enumerate(self.decoder_layers):
            token=decoder_layers(token,pos_cis)
        token=self.rmsnorm(token)
        

        if target is not None:

            logits = self.Linear(token)
            self.last_loss=F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1)  
        else:
            logits = self.Linear(token[:, [-1], :])  
            self.last_loss = None  


        self.OUT.__setitem__('logits', logits) 
        self.OUT.__setitem__('last_loss', self.last_loss)  
        

        return self.OUT
    
    
    @torch.inference_mode()
    def generate(self,token,eos):
        index = token.shape[1]
        max_new_tokens=50
        temperature  =0.7
        repetition_penalty=1.
        stream=True

        while index < max_new_tokens - 1:
           inference_res = self(token)
           logits = inference_res.logits
           logits = logits[:, -1, :]

           logits = logits / temperature
           probs = F.softmax(logits, dim=-1) 
           idx_next = torch.multinomial(probs, num_samples=1, generator=None)  
           if idx_next == eos:  
                break
           token = torch.cat((token, idx_next), dim=1)

        return token





        