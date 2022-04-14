from encoder import Encoder
from query_attn import Query_Self_Attn
from weight import Make_Weights
from decoder import Decoder
import torch.nn as nn
import torch
class Cond_Query_Tr(nn.Module):
    def __init__(self,image_size=(32,96,96),img_dim=256,tokens_dim=256,mlp_dim=512,layers=4,query_attn_layers=2,heads=8,num_query=32,num_cls=7,cat=True,dropout=0.1,drop_path_rate=0):
        super().__init__()
        self.image_size = image_size
        self.num_query = num_query
        self.num_cls = num_cls
        self.task_embed_organ = nn.Linear(num_cls,tokens_dim)
        self.task_embed_tumor = nn.Linear(num_cls,tokens_dim)
        if self.image_size[0]%16!=0 or self.image_size[1]%16!=0 or self.image_size[2]%16!=0:
            raise ValueError("image dimensions must be divisible by 16")
        self.spatial_size = (int(self.image_size[0]/16),int(self.image_size[1]/16),int(self.image_size[2]/16))
        self.query = nn.Parameter(torch.randn(1, num_query, tokens_dim))
        self.pos_embed = nn.Parameter(torch.randn(1,self.spatial_size[0]*self.spatial_size[1]*self.spatial_size[2],tokens_dim))
        self.encoder = Encoder(spatial_size=self.spatial_size,layers=layers,heads=heads,tokens_dim=tokens_dim,mlp_dim=mlp_dim,img_dim=img_dim,dropout=dropout,drop_path_rate=drop_path_rate)
        self.query_self_attn = Query_Self_Attn(tokens_dim=tokens_dim, mlp_dim=mlp_dim, heads=heads, layers=query_attn_layers, add_task_embed=True)
        self.make_weights = Make_Weights(img_dim=img_dim,tokens_dim=tokens_dim, num_query=num_query, spatial_size=self.spatial_size)
        self.decoder = Decoder(num_query=num_query,layers=layers,heads=heads,tokens_dim=tokens_dim,mlp_dim=mlp_dim,cat=cat,dropout=dropout,drop_path_rate=drop_path_rate)
    def forward(self,img_bottom,task_encoding):
        B,_,d,h,w = img_bottom.shape
        task_embed_organ = self.task_embed_organ(task_encoding)
        task_embed_tumor = self.task_embed_tumor(task_encoding)
        pos_embed = self.pos_embed.expand(B,-1,-1).cuda()
        cls_token,img_tokens = self.encoder(img_bottom,pos_embed)
        query = self.query.expand(B,-1,-1)


        query_attn_organ = self.query_self_attn(query,cls_token,task_embed_organ)
        query_result_organ = self.decoder(img_tokens,query_attn_organ,pos_embed)#(B,num_patches,dim),(B,num_query,dim)
        input_upsample_organ = self.make_weights(task_embed_organ,query_attn_organ,query_result_organ)

        query_attn_tumor = self.query_self_attn(query, cls_token, task_embed_tumor)
        query_result_tumor = self.decoder(img_tokens, query_attn_tumor,pos_embed)  # (B,num_patches,dim),(B,num_query,dim)
        input_upsample_tumor = self.make_weights(task_embed_tumor, query_attn_tumor, query_result_tumor)

        return input_upsample_organ,input_upsample_tumor#(B,2,D,H,W)




