import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class MlpBlock(nn.Module):
    """ Transformer Feed-Forward Block """
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):

        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)

        out = self.fc2(out)
        out = self.dropout2(out)
        return out


class LinearGeneral(nn.Module):
    def __init__(self, in_dim=(768,), feat_dim=(12, 64)):
        super(LinearGeneral, self).__init__()

        self.weight = nn.Parameter(torch.randn(*in_dim, *feat_dim))
        self.bias = nn.Parameter(torch.zeros(*feat_dim))

    def forward(self, x, dims):
        a = torch.tensordot(x, self.weight, dims=dims) + self.bias
        return a


class SelfAttention(nn.Module):
    def __init__(self, in_dim, heads=8, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_dim // heads
        self.scale = self.head_dim ** 0.5

        self.query = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.key = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.value = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.out = LinearGeneral((self.heads, self.head_dim), (in_dim,))

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        b, n, _ = x.shape

        q = self.query(x, dims=([2], [0]))
        k = self.key(x, dims=([2], [0]))
        v = self.value(x, dims=([2], [0]))

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3)

        out = self.out(out, dims=([2, 3], [0, 1]))

        return out

class EncoderBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, num_heads, dropout_rate=0.1, attn_dropout_rate=0.1):
        super(EncoderBlock, self).__init__()

        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = SelfAttention(in_dim, heads=num_heads, dropout_rate=attn_dropout_rate)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.attn(out)
        if self.dropout:
            out = self.dropout(out)
        out += residual
        residual = out

        out = self.norm2(out)
        out = self.mlp(out)
        out += residual
        return out

class VOEncoder(nn.Module):
    def __init__(self, image_num, des_num, des_dim, des_emb_dim, mlp_dim, num_layers=12, num_heads=12, dropout_rate=0.1, attn_dropout_rate=0.0, device='cuda'):
        super(VOEncoder, self).__init__()

        # positional embedding
        self.pos_embedding = VOPositionEmbs(image_num, des_num, des_dim, des_emb_dim, dropout_rate, device)

        # encoder blocks
        in_dim = des_emb_dim + 2 + 1

        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = EncoderBlock(in_dim, mlp_dim, num_heads, dropout_rate, attn_dropout_rate)
            self.encoder_layers.append(layer)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):

        out = self.pos_embedding(x)

        for layer in self.encoder_layers:
            out = layer(out)

        out = self.norm(out)
        # print(out.shape)
        return out

class VOPositionEmbs(nn.Module):
    def __init__(self, image_num, des_num, des_dim, des_emb_dim, dropout_rate=0.1, device='cuda'):
        super(VOPositionEmbs, self).__init__()
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

        self.des_embedding = nn.Linear(des_dim, des_emb_dim)
        self.des_dim = des_dim
        self.des_num = des_num
        self.image_num = image_num
        self.device = device

    def forward(self, x):
        # direct encode with frame id
        # pose = torch.ones(x.size(0), x.size(1), x.size(2), 1)
        # for j in range(self.image_num):
        #     pose[:, j, :, 0] = j
        # pose = pose.to(self.device)
        # out = torch.concat([pose, x], dim=3)
        # out = out.reshape(out.size(0), out.size(1) * out.size(2), out.size(3))

        # positional encoding
        pe = torch.ones(x.size(0), x.size(1), x.size(2), x.size(3))
        print(x.size())
        for j in range(self.image_num):
            d_model = self.des_dim + 2
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            div_term_ = torch.exp(torch.arange(0, d_model-1, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, j, :, 0::2] = torch.sin(j * div_term)
            pe[:, j, :, 1::2] = torch.cos(j * div_term_)
        # print(pe)
        pe = pe.to(self.device)
        out = x + pe
        out = out.reshape(out.size(0), out.size(1) * out.size(2), out.size(3))

        if self.dropout:
            out = self.dropout(out)

        return out

class VOTransformer(nn.Module):
    def __init__(self,
                 image_num=31,
                 des_num=128,
                 des_dim=27,
                 des_emb_dim=49,
                 mlp_dim=128,
                 num_heads=8,
                 num_layers=2,
                 attn_dropout_rate=0.1,
                 dropout_rate=0.0,
                 device='cuda'):
        super(VOTransformer, self).__init__()

        self.des_num = des_num
        self.des_emb_dim = des_emb_dim
        self.image_num = image_num

        self.transformer = VOEncoder(
            image_num=image_num,
            des_num=des_num,
            des_dim=des_dim,
            des_emb_dim=des_emb_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            device=device
        )

        self.image_num = image_num
        in_dim = des_num * (des_emb_dim + 2 + 1)

        self.linear1 = nn.Linear(in_dim, 256)
        self.linear2 = nn.Linear(image_num * 256, (image_num - 1) * 128)
        self.linear3 = nn.Linear((image_num - 1) * 128, (image_num - 1) * 32)
        self.linear4 = nn.Linear((image_num - 1) * 32, (image_num - 1) * 6)

    def forward(self, x):
        x = self.transformer(x)
        x = x.reshape(x.size(0), self.image_num, -1)

        x = self.linear1(x)
        x = x.reshape(x.size(0), -1)
        # res = self.shortcut(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        # x = res + x
        out = x.reshape(x.size(0), self.image_num - 1, -1)

        # separate
        # t = self.linear1_t(x)
        # t = t.reshape(t.size(0), -1)
        # t = self.linear2_t(t)
        # t = self.linear3_t(t)
        # t = t.reshape(t.size(0), self.image_num - 1, -1)
        #
        # r = self.linear1_r(x)
        # r = r.reshape(r.size(0), -1)
        # r = self.linear2_r(r)
        # r = self.linear3_r(r)
        # r = r.reshape(r.size(0), self.image_num - 1, -1)
        #
        # out = torch.concat([r, t], dim=2)

        return out

if __name__ == '__main__':
    model = VOTransformer(num_layers=4, device='cpu')
    x = torch.randn((1, 31, 128, 29))
    out = model(x)

    state_dict = model.state_dict()

    print(out.shape)

    for key, value in state_dict.items():
        print("{}: {}".format(key, value.shape))
