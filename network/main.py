import torch.nn as nn
from torch.nn import functional as F

import torch

affine_par = True

from cond_query_tr import Cond_Query_Tr

from utils_ import conv3x3x3, NoBottleneck

in_place = True


class skip_attn(nn.Module):
    def __init__(self, inplanes, weight_std):
        super(skip_attn, self).__init__()
        self.inplanes = inplanes
        self.weight_std = weight_std
        self.gn1 = nn.GroupNorm(16, 2 * inplanes)
        self.conv1 = conv3x3x3(in_planes=self.inplanes * 2, out_planes=self.inplanes, weight_std=self.weight_std)
        self.relu1 = nn.LeakyReLU(0.1, inplace=in_place)
        self.gn2 = nn.GroupNorm(16, inplanes)
        self.conv2 = conv3x3x3(in_planes=self.inplanes, out_planes=1, weight_std=self.weight_std)
        self.sigmoid = nn.Sigmoid()

    def forward(self, skip, x):
        skip_attn = torch.cat([skip, x], dim=1)
        skip_attn = self.relu1(self.conv1(self.gn1(skip_attn)))
        skip_attn = self.sigmoid(self.conv2(self.gn2(skip_attn)))
        skip = skip * skip_attn
        return skip


class unet3D(nn.Module):
    def __init__(self, conv_layers=[1, 2, 2, 2, 2], image_size=(32, 96, 96), img_dim=256, tokens_dim=256, mlp_dim=512,
                 img_attn_layers=4, query_attn_layers=2, heads=8, num_query=32, num_cls=7, dropout=0.1,
                 drop_path_rate=0, weight_std=True, skipattn=True, cat=True):
        super(unet3D, self).__init__()
        self.inplanes = 128
        self.weight_std = weight_std
        self.skipattn = skipattn
        self.cat = cat
        self.num_cls = num_cls
        self.conv1 = conv3x3x3(1, 32, stride=[1, 1, 1], weight_std=self.weight_std)

        self.layer0 = self._make_layer(NoBottleneck, 32, 32, conv_layers[0], stride=(1, 1, 1))
        self.layer1 = self._make_layer(NoBottleneck, 32, 64, conv_layers[1], stride=(2, 2, 2))
        self.layer2 = self._make_layer(NoBottleneck, 64, 128, conv_layers[2], stride=(2, 2, 2))
        self.layer3 = self._make_layer(NoBottleneck, 128, 256, conv_layers[3], stride=(2, 2, 2))
        self.layer4 = self._make_layer(NoBottleneck, 256, 256, conv_layers[4], stride=(2, 2, 2))

        self.fusionConv = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(0.1, inplace=in_place),
            conv3x3x3(256, 256, kernel_size=(1, 1, 1), padding=(0, 0, 0), weight_std=self.weight_std)
        )
        self.GAP = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=in_place),
            torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.controller = nn.Conv3d(256 + self.num_cls, 144, kernel_size=1, stride=1, padding=0)
        self.contro_organ = nn.Conv3d(256 + self.num_cls, 9, kernel_size=1, stride=1, padding=0)
        self.contro_tumor = nn.Conv3d(256 + self.num_cls, 9, kernel_size=1, stride=1, padding=0)
        self.precls_conv = nn.Sequential(
            nn.GroupNorm(16, 32),
            nn.LeakyReLU(0.1, inplace=in_place),
            nn.Conv3d(32, 8, kernel_size=1)
        )
        self.CCQ = Cond_Query_Tr(image_size=image_size, img_dim=img_dim, tokens_dim=tokens_dim, mlp_dim=mlp_dim,
                                  layers=img_attn_layers, query_attn_layers=query_attn_layers, heads=heads,
                                  num_query=num_query, num_cls=num_cls, cat=self.cat, dropout=dropout,
                                  drop_path_rate=drop_path_rate)

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.skip3_attn = skip_attn(256, self.weight_std)
        self.x8_resb = self._make_layer(NoBottleneck, 256, 128, 1, stride=(1, 1, 1))
        self.skip2_attn = skip_attn(128, self.weight_std)
        self.x4_resb = self._make_layer(NoBottleneck, 128, 64, 1, stride=(1, 1, 1))
        self.skip1_attn = skip_attn(64, self.weight_std)
        self.x2_resb = self._make_layer(NoBottleneck, 64, 32, 1, stride=(1, 1, 1))
        self.skip0_attn = skip_attn(32, self.weight_std)
        self.x1_resb = self._make_layer(NoBottleneck, 32, 32, 1, stride=(1, 1, 1))
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, inplanes, planes, blocks, stride=(1, 1, 1), dilation=1, multi_grid=1):
        downsample = None

        if stride[0] != 1 or stride[1] != 1 or stride[2] != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.GroupNorm(16, inplanes),
                nn.LeakyReLU(0.1, inplace=in_place),
                conv3x3x3(inplanes, planes, kernel_size=(1, 1, 1), stride=stride, padding=0,
                          weight_std=self.weight_std),
            )

        layers = []

        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std))

        for i in range(1, blocks):
            layers.append(

                block(planes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                      weight_std=self.weight_std))

        return nn.Sequential(*layers)

    def encoding_task(self, task_id):
        N = task_id.shape[0]
        task_encoding = torch.zeros(size=(N, self.num_cls))
        for i in range(N):
            task_encoding[i, task_id[i]] = 1
        return task_encoding.cuda()

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 1)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 5
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv3d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.leaky_relu(x, 0.1, inplace=in_place)
        return x

    def forward(self, input, task_id):
        x = self.conv1(input)
        x = self.layer0(x)
        skip0 = x  # 32

        x = self.layer1(x)
        skip1 = x  # 64

        x = self.layer2(x)
        skip2 = x  # 128

        x = self.layer3(x)
        skip3 = x  # 256

        x = self.layer4(x)

        x = self.fusionConv(x)

        ######
        task_encoding = self.encoding_task(task_id)
        ######
        #######11111111
        organ, tumor = self.CCQ(x, task_encoding[:, None, :])
        #######
        #######
        x_feat = self.GAP(x)
        x_cond = torch.cat([x_feat, task_encoding[:, :, None, None, None]], 1)
        params, params_organ, params_tumor = self.controller(x_cond), self.contro_organ(x_cond), self.contro_tumor(
            x_cond)
        params.squeeze_(-1).squeeze_(-1).squeeze_(-1)
        params_organ.squeeze_(-1).squeeze_(-1).squeeze_(-1)
        params_tumor.squeeze_(-1).squeeze_(-1).squeeze_(-1)
        #######
        #####------------------organ_upsample-----------------------
        # x8
        organ = self.upsamplex2(organ)
        if self.skipattn:
            skip3organ = self.skip3_attn(skip3, organ)
            organ = organ + skip3organ
        else:
            organ = organ + skip3
        organ = self.x8_resb(organ)

        # x4
        organ = self.upsamplex2(organ)
        if self.skipattn:
            skip2organ = self.skip2_attn(skip2, organ)
            organ = organ + skip2organ
        else:
            organ = organ + skip2
        organ = self.x4_resb(organ)

        # x2
        organ = self.upsamplex2(organ)
        if self.skipattn:
            skip1organ = self.skip1_attn(skip1, organ)
            organ = organ + skip1organ
        else:
            organ = organ + skip1
        organ = self.x2_resb(organ)

        # x1
        organ = self.upsamplex2(organ)
        if self.skipattn:
            skip0organ = self.skip0_attn(skip0, organ)
            organ = organ + skip0organ
        else:
            organ = organ + skip0
        organ = self.x1_resb(organ)
        head_inputs_organ = self.precls_conv(organ)
        # --------conditonal conv-----------------------------
        N, _, D, H, W = head_inputs_organ.size()
        weight_nums, bias_nums = [8 * 8, 8 * 8, 8 * 1], [8, 8, 1]
        # ------------------organ_cond_conv-------------------------
        head_inputs_organ = head_inputs_organ.reshape(1, -1, D, H, W)
        weights_organ, biases_organ = self.parse_dynamic_params(torch.cat((params, params_organ), dim=1), 8,
                                                                weight_nums, bias_nums)
        organ = self.heads_forward(head_inputs_organ, weights_organ, biases_organ, N).reshape(-1, 1, D, H, W)
        # ---------------------------------------------------------
        # ------------------tumor_upsample-----------------------
        # x8
        tumor = self.upsamplex2(tumor)
        if self.skipattn:
            skip3tumor = self.skip3_attn(skip3, tumor)
            tumor = tumor + skip3tumor
        else:
            tumor = tumor + skip3
        tumor = self.x8_resb(tumor)

        # x4
        tumor = self.upsamplex2(tumor)
        if self.skipattn:
            skip2tumor = self.skip2_attn(skip2, tumor)
            tumor = tumor + skip2tumor
        else:
            tumor = tumor + skip2
        tumor = self.x4_resb(tumor)

        # x2
        tumor = self.upsamplex2(tumor)
        if self.skipattn:
            skip1tumor = self.skip1_attn(skip1, tumor)
            tumor = tumor + skip1tumor
        else:
            tumor = tumor + skip1
        tumor = self.x2_resb(tumor)

        # x1
        tumor = self.upsamplex2(tumor)
        if self.skipattn:
            skip0tumor = self.skip0_attn(skip0, tumor)
            tumor = tumor + skip0tumor
        else:
            tumor = tumor + skip0
        tumor = self.x1_resb(tumor)
        head_inputs_tumor = self.precls_conv(tumor)
        # -------------------tumor_cond_conv---------------------
        head_inputs_tumor = head_inputs_tumor.reshape(1, -1, D, H, W)
        weights_tumor, biases_tumor = self.parse_dynamic_params(torch.cat((params, params_tumor), dim=1), 8,
                                                                weight_nums, bias_nums)
        tumor = self.heads_forward(head_inputs_tumor, weights_tumor, biases_tumor, N).reshape(-1, 1, D, H, W)
        # ----------------------------------------------
        return self.sigmoid(torch.cat((organ, tumor), dim=1))
