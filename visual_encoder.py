import torch
import torch.nn as nn
from transformer_layer import HierarchyPositionalEncoding

class Visual_Encoder(nn.Module):
    def __init__(self, v_img_size, v_patch_size, patch_out_d, out_d, img_num, dropout=0.1, device='cpu'):
        super(Visual_Encoder, self).__init__()
        assert v_img_size % v_patch_size == 0, 'img size must be divisible by patch size!'

        self.v_patch_num = (v_img_size // v_patch_size) ** 2
        self.v_patch_size = v_patch_size
        self.img_num = img_num
        self.patch_out_d = patch_out_d
        
        v_patch_dim = 3 * v_patch_size * v_patch_size
        self.v_to_patch = nn.Sequential(
                nn.LayerNorm(v_patch_dim),
                nn.Linear(v_patch_dim, patch_out_d),
                nn.LayerNorm(patch_out_d)
                )
        
        self.proj_v = nn.Linear(img_num*self.v_patch_num*patch_out_d, out_d)
        
        # hierarchical positional encoding
        self.pos_v = HierarchyPositionalEncoding(img_num, self.v_patch_num, patch_out_d, device)
        self.dropout = nn.Dropout(dropout)
  
    def forward(self, v):
        v = v.reshape(-1, self.img_num, self.v_patch_num, 3 * self.v_patch_size * self.v_patch_size)
        v = self.v_to_patch(v)
        v += self.pos_v.to(v.device)
        v = v.reshape(-1, self.img_num * self.v_patch_num * self.patch_out_d)
        v = self.dropout(v)
        v = self.proj_v(v)
        return v

'''
model = Visual_Encoder(224, 4, 2, 1024, 15)
v = torch.randn(10, 15, 3, 224, 224)
print(model(v).shape)
'''