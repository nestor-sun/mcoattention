import torch
import torch.nn as nn
from multimodal_coattention_layer import Multimodal_Coattention
from visual_encoder import Visual_Encoder

class PersonalityModel(nn.Module):
    def __init__(self, a_dim, t_dim, v_img_size, v_patch_size, heads, patch_out_d, coa_d, \
			out_d, img_num, dropout=0.1, device='cpu'):
        super(PersonalityModel, self).__init__()

        # modal linear projection 
        self.proj_v = nn.Sequential(*[Visual_Encoder(v_img_size, v_patch_size, patch_out_d, coa_d, img_num, dropout, device), \
                                      nn.GELU(), nn.Dropout(dropout)])
        self.proj_a = nn.Sequential(*[nn.Linear(a_dim, coa_d), nn.GELU(), nn.Dropout(dropout)])
        self.proj_t = nn.Sequential(*[nn.Linear(t_dim, coa_d), nn.GELU(), nn.Dropout(dropout)])

        self.coattention = Multimodal_Coattention(heads, coa_d, 3, dropout)
        self.fc = nn.Linear(coa_d, out_d)

    def forward(self, a, v, t):
        a = self.proj_a(a)
        t = self.proj_t(t)
        v = self.proj_v(v)

        combined = torch.stack((a,v,t), dim=1)	
        combined = self.coattention(combined)
        out = self.fc(combined)
	
        return out.sigmoid()

'''
model = PersonalityModel(234, 768, 224, 4, 8, 2, 1024, 1, 15)
a = torch.randn(10, 234)
t = torch.randn(10, 768)
v = torch.randn(10, 15, 3, 224, 224)
print(model(a,v,t))
'''
