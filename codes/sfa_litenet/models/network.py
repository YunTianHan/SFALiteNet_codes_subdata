"""四级轻量编码器、SFF 瓶颈和三级 LAD 解码器。"""

import torch
from torch import nn

from .modules import DSC, LAD, ResidualDSC, SpatialFrequencyFusion


class SFALiteNet(nn.Module):
    """输入 B×3×H×W，输出 B×1×H×W 的未激活 logits。

    默认 512 输入的编码尺寸为 512/256/128/64，瓶颈保持 64；
    三级解码恢复至 128/256/512。训练直接使用 logits 以稳定计算 Focal。
    概率接口显式施加 sigmoid。
    """

    def __init__(self, config):
        super().__init__()
        channels = config["encoder_channels"]
        incoming = 3
        self.encoder = nn.ModuleList()
        for index, outgoing in enumerate(channels):
            # 第一阶段只使用 DSC，后三阶段使用带步幅 2 的投影残差块。
            blocks = [DSC(incoming, outgoing)] if index == 0 else [ResidualDSC(incoming, outgoing, stride=2)]
            blocks.extend(ResidualDSC(outgoing, outgoing) for _ in range(config["blocks_per_stage"]-1))
            self.encoder.append(nn.Sequential(*blocks))
            incoming = outgoing
        bottleneck = config["bottleneck_channels"]
        self.project = nn.Sequential(nn.Conv2d(channels[-1], bottleneck, 1, bias=False),
                                     nn.BatchNorm2d(bottleneck), nn.SiLU(inplace=True))
        self.sff = SpatialFrequencyFusion(bottleneck, config["frequency_reduction"], config["eca_kernel"])
        self.decoder = nn.ModuleList()
        incoming = bottleneck
        for skip_channels in reversed(channels[:3]):
            self.decoder.append(LAD(incoming, skip_channels, skip_channels, config))
            incoming = skip_channels
        self.head = nn.Conv2d(incoming, 1, 1)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("模型输入必须为 B×3×H×W RGB 张量")
        if min(x.shape[-2:]) < 16:
            raise ValueError("模型输入高宽不得小于 16")
        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
        x = self.sff(self.project(skips[-1]))
        for block, skip in zip(self.decoder, reversed(skips[:3])):
            x = block(x, skip)
        return self.head(x)

    @torch.no_grad()
    def predict_proba(self, x):
        """调用者应先设置 eval；该接口不改变模型原有训练状态。"""
        return self(x).sigmoid()
