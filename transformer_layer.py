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
    #print(patch_encoding.shape, sequence_encoding.shape)
    
    patch_encoding = patch_encoding.unsqueeze(0).repeat(video_length, 1, 1)
    sequence_encoding = sequence_encoding.unsqueeze(1).repeat(1, patch_num, 1)
    #print(patch_encoding.shape, sequence_encoding.shape)
    encoding = patch_encoding + sequence_encoding
    return encoding


#pe = HierarchyPositionalEncoding(15, 256, 48, 'cpu')





