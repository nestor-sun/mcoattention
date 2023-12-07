import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
import numpy as np
import random
from torch.autograd import Variable


class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_old, x_new):
        return self.norm(self.dropout(x_new) + x_old)


def attention_matmul(q,k,v):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)/math.sqrt(d_k))
    #print('QK dimension', scores.shape)
    attention = F.softmax(scores, dim=-1)
    return torch.matmul(attention, v), attention


class MultiHeadAttention(nn.Module):
    def __init__(self, num_of_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_of_heads == 0
        self.d_k = int(d_model/num_of_heads)
        self.num_of_heads = num_of_heads
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])

    def forward(self, query, key, value):
        num_of_batches = query.size(0)
        query, key, value = [linear(x).view(num_of_batches, -1, self.num_of_heads, self.d_k).transpose(1,2)\
                                    for linear, x in zip(self.linears, (query, key, value))]
        x, attention = attention_matmul(query, key, value)
        x = x.transpose(1, 2).contiguous().view(num_of_batches, self.num_of_heads*self.d_k)

        return x, attention


class SelfAttention(nn.Module):
    def __init__(self, num_of_heads, d, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.attention = MultiHeadAttention(num_of_heads, d)
        self.norm1 = AddNorm(d, dropout)
        self.linear = nn.Linear(d, d)
        self.norm2 = AddNorm(d, dropout)

    def forward(self, i):
        x, attention = self.attention(i, i, i)
        x_norm = self.norm1(i, x)
        x_linear = self.linear(x_norm)
        x_normed = self.norm2(x_norm, x_linear)
        return x_normed


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class
        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size = x.size()[0]
        seq_len = x.size()[1]
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]


def HierarchyPositionalEncoding(video_length, patch_num, d_model, device):
    patch_encoding = torch.zeros((patch_num, d_model), device=device)
    sequence_encoding = torch.zeros((video_length, d_model), device=device)

    patch_pos = torch.arange(0, patch_num, device=device)
    sequence_pos = torch.arange(0, video_length, device=device)

    patch_pos = patch_pos.float().unsqueeze(dim=1)
    sequence_pos = sequence_pos.float().unsqueeze(dim=1)
    # 1D => 2D unsqueeze to represent word's position   

    patch_2i = torch.arange(0, d_model, step=2, device=device)
    sequence_2i = torch.arange(0, d_model, step=2, device=device)

    patch_encoding[:, 0::2] = torch.sin(patch_pos / (10000 ** (patch_2i / d_model)))
    patch_encoding[:, 1::2] = torch.cos(patch_pos / (10000 ** (patch_2i / d_model)))

    sequence_encoding[:, 0::2] = torch.sin(sequence_pos / (10000 ** (sequence_2i / d_model)))
    sequence_encoding[:, 1::2] = torch.cos(sequence_pos / (10000 ** (sequence_2i / d_model)))
    
    patch_encoding = patch_encoding.unsqueeze(0).repeat(video_length, 1, 1)
    sequence_encoding = sequence_encoding.unsqueeze(1).repeat(1, patch_num, 1)
    encoding = patch_encoding + sequence_encoding
    return encoding


class Coattention(nn.Module):
    def __init__(self, heads_num, d, dropout=0.1):
        super(Coattention, self).__init__()
        self.attention = MultiHeadAttention(heads_num, d)
        self.coattention = MultiHeadAttention(heads_num, d)
        
        self.attention_norm = AddNorm(d, dropout)
        self.attention_linear = nn.Linear(d, d)
        
        self.coattention_norm = AddNorm(d, dropout)
        self.coattention_linear = nn.Linear(d,d)

    def forward(self, three_inputs):
        m1 = three_inputs[0]
        m2 = three_inputs[1]
        m3 = three_inputs[2]
        m1_new, _ = self.attention(m1,m1,m1)
        m1 = self.attention_norm(m1_new, m1)
        m1 = self.attention_linear(m1)

        c = torch.stack([m2, m3], dim=1)
        x, coattention = self.coattention(m1, c, c)
        x_normed = self.coattention_norm(x, m1)
        x = self.coattention_linear(x_normed)
        return x, coattention


class MultiModalCoattention(nn.Module):
    def __init__(self, a_img_size, a_patch_size, v_img_size, v_patch_size, heads, patch_out_d, coa_d, coattention_layer_num, video_len, img_num, dropout=0.1, device='cpu', v_n_channel=3, a_n_channel=3, a_dim=219544, text_dim=4362, if_no_pos_enc=False, debug=False, label='agreeableness'):
        super(MultiModalCoattention, self).__init__()
        assert a_img_size % a_patch_size == 0 and v_img_size % v_patch_size == 0, 'img size must be divisible by patch size!'
        
        self.v_n_channel = v_n_channel
        self.a_n_channel = a_n_channel

        self.a_patch_num = (a_img_size // a_patch_size) ** 2
        self.v_patch_num = (v_img_size // v_patch_size) ** 2
        self.a_patch_size = a_patch_size
        self.v_patch_size = v_patch_size
        self.video_len = video_len
        self.img_num = img_num
        self.patch_out_d = patch_out_d

        self.debug=debug

        av_layers = []
        at_layers = []
        vt_layers = []
        for i in range(coattention_layer_num):
            av = Coattention(heads, coa_d, dropout)
            at = Coattention(heads, coa_d, dropout)
            vt = Coattention(heads, coa_d, dropout)

            av_layers.append(av)
            at_layers.append(at)
            vt_layers.append(vt)
    
        self.av = nn.Sequential(*av_layers)
        self.at = nn.Sequential(*at_layers)
        self.vt = nn.Sequential(*vt_layers)
        
        # modal linear projection 
        v_patch_dim = v_n_channel * v_patch_size * v_patch_size
        self.v_to_patch = nn.Sequential(
            nn.LayerNorm(v_patch_dim),
            nn.Linear(v_patch_dim, patch_out_d),
            nn.LayerNorm(patch_out_d)
            )
        
        # hierarchical positional encoding
        self.if_no_pos_enc = if_no_pos_enc
        if not if_no_pos_enc:
            self.pos_a = HierarchyPositionalEncoding(video_len, self.a_patch_num, patch_out_d, device)
            self.pos_v = HierarchyPositionalEncoding(img_num, self.v_patch_num, patch_out_d, device)

        self.dropout = nn.Dropout(dropout)
        #print('Text dim', text_dim) 
        self.proj_v = nn.Linear(img_num*self.v_patch_num*patch_out_d, coa_d)
        self.proj_t = nn.Linear(text_dim, coa_d)
        self.proj_a = nn.Linear(a_dim, coa_d)
        self.gelu = nn.GELU()

        fc = []
        fc.append(nn.Linear(coa_d+coa_d+coa_d, coa_d))
        fc.append(nn.GELU())
        if label == 'all' or label == 'all_bi':
            fc.append(nn.Linear(coa_d, 5))
        else:
            fc.append(nn.Linear(coa_d, 1))
        fc.append(nn.Sigmoid())
        self.fc = nn.Sequential(*fc)
 
    def forward(self, a, v, t):
        v = v.reshape(-1, self.img_num, self.v_patch_num, self.v_n_channel * self.v_patch_size * self.v_patch_size)
        v = self.v_to_patch(v)
        
        if not self.if_no_pos_enc:
            v += self.pos_v.to(v.device)
         
        v = v.reshape(-1, self.img_num * self.v_patch_num * self.patch_out_d)
         
        a = self.dropout(a)
        v = self.dropout(v)
        t = self.dropout(t)

        a = self.proj_a(a)
        v = self.proj_v(v)
        t = self.proj_t(t)
        
        a = self.gelu(a)
        v = self.gelu(v)
        t = self.gelu(t)

        a = self.dropout(a)
        v = self.dropout(v)
        t = self.dropout(t)

        for av, vt, at in zip(self.av, self.vt, self.at): 
            t, t_co = av((t, v, a))
            a, a_co = vt((a, v, t))
            v, v_co = at((v, a, t))

        combined = torch.cat((a, v, t), dim=1)
        out = self.fc(combined)
        return out

# Example
'''
model = MultiModalCoattention(64, 2, 64, 2, 16, 16, 512, 1, 15, 100, 0.0, 'cuda', 3, 1, 219544, 8095, False, False, 'agreeableness')
v = torch.randn(6, 100, 3, 64, 64)
a = torch.randn(6, 219544)
t = torch.randn(6, 8095)
out = model(a, v, t)
print(out.shape)
'''
