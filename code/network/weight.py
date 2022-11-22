import torch
import torch.nn as nn
from utils_ import conv3x3x3
in_place=True
class Make_Weights(nn.Module):
    def __init__(self,img_dim,tokens_dim,num_query,spatial_size):
        super().__init__()

        self.num_query = num_query
        self.d,self.h,self.w=spatial_size
        self.img_dim=img_dim
        self.proj1 = nn.Sequential(nn.LayerNorm(tokens_dim),nn.Linear(tokens_dim,tokens_dim),nn.LeakyReLU(0.1,inplace=in_place))
        self.proj2 = nn.Sequential(nn.LayerNorm(tokens_dim),nn.Linear(tokens_dim,tokens_dim),nn.LeakyReLU(0.1,inplace=in_place))
        self.proj3 = nn.Sequential(nn.LayerNorm(2*tokens_dim),nn.Linear(2*tokens_dim,tokens_dim),nn.LeakyReLU(0.1,inplace=in_place))
        self.proj4 = nn.Sequential(nn.LayerNorm(2 * tokens_dim), nn.Linear(2 * tokens_dim, 1), nn.Sigmoid())
        self.proj5 = nn.Sequential(nn.LayerNorm(tokens_dim), nn.Linear(tokens_dim, self.d * self.h * self.w),
                                   nn.LeakyReLU(0.1, inplace=in_place))
        self.conv = conv3x3x3(self.num_query, self.img_dim)
    def forward(self,task_encoding,query_attn,query_result):
        B = query_attn.shape[0]
        task_encoding = task_encoding.expand(-1, self.num_query, -1)
        query_attn_feat = self.proj1(query_attn)
        query_result_feat = self.proj2(query_result)
        query_feat = torch.cat([query_attn_feat, query_result_feat], dim=2)
        query_feat = self.proj3(query_feat)

        query_feat = torch.cat([query_feat, task_encoding], dim=2)
        confidence = self.proj4(query_feat)
        weight = confidence * query_result
        weight = self.proj5(weight)
        weight = weight.view((B, self.num_query, self.d, self.h, self.w))

        input_upsample = self.conv(weight)

        return input_upsample  # (1, 256, 2, 6, 6)

