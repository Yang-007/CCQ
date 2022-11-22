import torch.nn  as nn
import torch
from tr_blocks import Blocks
from utils_ import init_weights


class Decoder(nn.Module):
    def __init__(self,num_query=32,layers=2,heads=8,tokens_dim=256,mlp_dim=512,cat=True,dropout=0.1,drop_path_rate=0):
        super().__init__()
        self.cat = cat
        self.scale = tokens_dim ** -0.5
        self.num_query = num_query
        self.blocks = Blocks(dim=tokens_dim, heads=heads, mlp_dim=mlp_dim, dropout=dropout, layers=layers,
                             drop_path_rate=drop_path_rate)
        self.proj_dec = nn.Linear(tokens_dim, tokens_dim)
        self.proj_query = nn.Parameter(self.scale * torch.randn(tokens_dim, tokens_dim))
        self.decoder_norm = nn.LayerNorm(tokens_dim)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self,img_tokens,query_attn,pos_embed):#img_tokens(B,dim,d,h,w),query)_attn:(B,num_query ,dim)
        img_tokens = self.proj_dec(img_tokens)
        if self.cat:
            img_tokens = torch.cat((query_attn,img_tokens), 1)#(B,N+num_query ,dim)
        query = self.blocks([query_attn,img_tokens,img_tokens],pos_embed)[0]
        query = self.decoder_norm(query)@ self.proj_query
        query_result = query / query.norm(dim=-1, keepdim=True)
        return query_result
