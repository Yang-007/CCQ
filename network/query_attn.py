import torch.nn  as nn
import torch
from tr_blocks import Blocks

class Query_Self_Attn(nn.Module):
    def __init__(self,tokens_dim=256,mlp_dim=512,heads=8,layers=4,add_task_embed=True):
        super().__init__()
        self.blocks=Blocks(dim=tokens_dim, heads=heads, mlp_dim=mlp_dim, layers=layers,dropout=0.1,drop_path_rate=0)
        self.add_task_embed=add_task_embed
    def forward(self,query,cls_token,task_encoding):#query:(B,32,256),cls_token:(b,1,256),task_encoding:(B,1,256)
        query_atten_feat=torch.cat((task_encoding,query),dim=1)
        if self.add_task_embed:

            query_atten_feat= self.blocks(torch.cat((cls_token,query_atten_feat),dim=1))[:,2:,:]
        else:
            query_atten_feat= self.blocks(query_atten_feat)[:,1:,:]
        return query_atten_feat#(1, 32, 256)





