import torch
import torch.nn as nn
from torchvision import models
from transformer_layer import *


class MultiHeadCoattention(nn.Module):
    def __init__(self, num_of_heads, d_model, modality_num):
        super(MultiHeadCoattention, self).__init__()
        assert d_model % num_of_heads == 0
        self.d_k = int(d_model/num_of_heads)
        self.num_of_heads = num_of_heads
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.modality_num = modality_num

    def forward(self, query, key, value):
        num_of_batches = query.size(0)
        query, key, value = [linear(x).view(num_of_batches, -1, self.num_of_heads, self.d_k).transpose(1,2)\
                            for linear, x in zip(self.linears, (query, key, value))]
        x, attention = attention_matmul(query, key, value)
        #print(x.shape)
        x = x.transpose(1, 2).contiguous().view(num_of_batches, self.modality_num, self.num_of_heads*self.d_k)
        return x


class Coattention(nn.Module):
    def __init__(self, heads, d, dropout, modality_num):
        super(Coattention, self).__init__()
        self.coattention = MultiHeadCoattention(heads, d, modality_num)
        self.norm = AddNorm(d, dropout)
        self.linear = nn.Linear(d,d)

    def forward(self, inputs):
        out = self.coattention(inputs, inputs, inputs)
        out = self.norm(inputs, out)
        out_linear = self.linear(out)
        out = self.norm(out, out_linear)
        return out


class Multimodal_Coattention(nn.Module):
    def __init__(self, heads, d, modality_num, dropout: float =0.1):
        super(Multimodal_Coattention, self).__init__()
        self.attentions = nn.ModuleList([SelfAttention(heads, d, dropout) for _ in range(modality_num)])
        self.coattention = Coattention(heads, d, dropout, modality_num)
        self.linear = nn.Linear(modality_num*d, d) 

    def forward(self, inputs):
        out = []
        for m_num, attention in enumerate(self.attentions):
            out.append(attention(inputs[:,m_num,:]))
        
        out = torch.stack(out, dim=1)
        out = self.coattention(out).reshape(out.shape[0], -1)
        out = self.linear(out)
        return out

'''
model = Multimodal_Coattention(4, 1024, 3, 0.1)
a = torch.randn(64, 3, 1024)
print(model(a).shape)
'''























