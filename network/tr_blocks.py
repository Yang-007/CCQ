import torch
import torch.nn as nn


import torch.nn.functional as F

from timm.models.layers import DropPath

in_place=True
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.LeakyReLU(0.1,inplace=in_place)
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None
        self.normq = nn.LayerNorm(dim)
        self.normk = nn.LayerNorm(dim)
        self.normv = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, q,k,v):
        B, Nq, C = q.shape
        Nk, Nv = k.shape[1], v.shape[1]
        q, k, v = self.normq(q), self.normk(k), self.normv(v)
        q = self.q(q).reshape(B, Nq, self.heads, C // self.heads).permute(0, 2, 1, 3)
        k = self.k(k).reshape(B, Nk, self.heads, C // self.heads).permute(0, 2, 1, 3)
        v = self.v(v).reshape(B, Nv, self.heads, C // self.heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        q = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        q = self.proj(q)
        q = self.proj_drop(q)
        return q,k,v


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        #self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x,pos_embed=None):
        if isinstance(x,list):
            q,k,v=x
        else:
            q=k=v=x
        if pos_embed is not None:
            if k.shape[1] == pos_embed.shape[1]:
                k, v = k + pos_embed, v + pos_embed
            elif k.shape[1] > pos_embed.shape[1]:
                start = q.shape[1]
                k_pos = k[:, start:, :] + pos_embed
                k = torch.cat((k[:, :start, :], k_pos), dim=1)
                v_pos = v[:, start:, :] + pos_embed
                v = torch.cat((v[:, :start, :], v_pos), dim=1)
            else:
                ValueError("k.shape[1] should >= pos_embed.shape[1]")

        temp1 = self.attn(q, k, v)[0]
        temp2 = self.drop_path(temp1)
        q = q + temp2
        q = q + self.drop_path(self.mlp(self.norm(q)))
        if isinstance(x, list):
            return [q, x[1], x[2]]
        else:
            return q

class Blocks(nn.Module):
    def __init__(self, dim, heads, mlp_dim, layers,dropout,drop_path_rate):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]
        self.layers=layers
        self.blocks =nn.ModuleList([Block(dim, heads, mlp_dim, dropout, dpr[i]) for i in range(layers)])
    def forward(self, x,pos_embed=None):

        for block in self.blocks:
            x=block(x,pos_embed)
        return x

