import torch.nn as nn
import torch
from tr_blocks import Blocks
from timm.models.layers import trunc_normal_
from utils_ import init_weights

# (B,embed_dim,image_size[0] // patch_size,image_size[1] // patch_size)->(B,embed_dim,num_patches)->(B,num_patches,embed_dim)
class Encoder(nn.Module):
    def __init__(self,spatial_size,layers=4,heads=8,tokens_dim=256,mlp_dim=512,img_dim=256,dropout=0.1,drop_path_rate=0.0):
        super().__init__()
        self.d, self.h, self.w = spatial_size
        self.img_dim = img_dim
        self.img_proj = nn.Linear(img_dim,tokens_dim)

        self.dropout = nn.Dropout(dropout)
        self.num_patches = self.d*self.h*self.w

        self.cls_token = nn.Parameter(torch.zeros(1, 1, tokens_dim))
        #self.distilled = distilled
        self.pos_embed = nn.Parameter(torch.randn(1, 1, tokens_dim))
        # transformer blocks
        self.blocks = Blocks(dim=tokens_dim, heads=heads, mlp_dim=mlp_dim, layers=layers,dropout=dropout,drop_path_rate=drop_path_rate)


        self.norm = nn.LayerNorm(tokens_dim)


        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)


        self.apply(init_weights)


    def forward(self, img_bottom,pos_embed): #B, img_dim, d, h, w
        B, _, d,h, w = img_bottom.shape
        x = img_bottom.flatten(2).permute(0,2,1)

        x = self.img_proj(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_pos_embed = self.pos_embed.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = torch.cat((cls_pos_embed,pos_embed),dim=1)

        x = self.dropout(x)
        x = self.blocks(x,pos_embed)
        x = self.norm(x)
        cls_token, img_tokens = x[:,0:1,:],x[:,1:,:]
        return cls_token,img_tokens




    

