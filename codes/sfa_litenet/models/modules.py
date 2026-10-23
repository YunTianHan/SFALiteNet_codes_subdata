"""轻量卷积、正交 Haar、能量门控及解码注意力。"""

import torch
from torch import nn
from torch.nn import functional as F


class DSC(nn.Sequential):
    """深度卷积 → 逐点卷积 → BN → SiLU；输入输出均为 BCHW。"""

    def __init__(self, incoming, outgoing, stride=1):
        super().__init__(
            nn.Conv2d(incoming, incoming, 3, stride, 1, groups=incoming, bias=False),
            nn.Conv2d(incoming, outgoing, 1, bias=False),
            nn.BatchNorm2d(outgoing), nn.SiLU(inplace=True),
        )


class ResidualDSC(nn.Module):
    """通道或步幅改变时投影残差，使四个编码阶段保持接口一致。"""

    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.main = DSC(incoming, outgoing, stride)
        self.shortcut = nn.Identity() if incoming == outgoing and stride == 1 else nn.Sequential(
            nn.Conv2d(incoming, outgoing, 1, stride, bias=False), nn.BatchNorm2d(outgoing)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(self.main(x) + self.shortcut(x))


def haar_dwt(x):
    """正交 Haar 分解，返回 B×4×C×ceil(H/2)×ceil(W/2) 与原尺寸。

    子带顺序为 LL、LH、HL、HH；L/H 第一字母对应竖直轴。
    h=[1,1]/sqrt(2)，g=[1,-1]/sqrt(2)，二维系数因此除以 2。
    奇数尺寸只在右边/底边复制一个像素，逆变换时裁掉补边。
    """
    size = x.shape[-2:]
    x = F.pad(x, (0, size[1] % 2, 0, size[0] % 2), mode="replicate")
    a, b, c, d = x[..., ::2, ::2], x[..., ::2, 1::2], x[..., 1::2, ::2], x[..., 1::2, 1::2]
    return torch.stack(((a+b+c+d)/2, (a-b+c-d)/2,
                        (a+b-c-d)/2, (a-b-c+d)/2), dim=1), size


def haar_iwt(bands, size):
    """将四个子带准确交错回 BCHW；保留梯度以训练能量门控。"""
    ll, lh, hl, hh = bands.unbind(dim=1)
    a, b = (ll+lh+hl+hh)/2, (ll-lh+hl-hh)/2
    c, d = (ll+lh-hl-hh)/2, (ll-lh-hl+hh)/2
    top = torch.stack((a, b), dim=-1).flatten(-2)
    bottom = torch.stack((c, d), dim=-1).flatten(-2)
    output = torch.stack((top, bottom), dim=-2).flatten(-3, -2)
    return output[..., :size[0], :size[1]]


class ECA(nn.Module):
    """全局池化后在通道轴执行一维卷积，不显式降维。"""

    def __init__(self, kernel):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel, padding=kernel//2, bias=False)

    def forward(self, x):
        descriptor = x.mean(dim=(-2, -1)).unsqueeze(1)
        weights = self.conv(descriptor).sigmoid().squeeze(1)[..., None, None]
        return x * weights


class SpatialFrequencyFusion(nn.Module):
    """融合空间特征与能量加权的 Haar 频率特征，并加入输入残差。"""

    def __init__(self, channels, reduction, eca_kernel):
        super().__init__()
        self.spatial = nn.Sequential(DSC(channels, channels), ECA(eca_kernel))
        hidden = max(1, channels//reduction)
        # Linear 自动作用于最后一个维度，同一 MLP 共享给全部四个子带。
        self.energy_mlp = nn.Sequential(nn.Linear(channels, hidden), nn.SiLU(), nn.Linear(hidden, channels))
        self.fuse = nn.Conv2d(2*channels, channels, 1, bias=False)

    def forward(self, x):
        bands, size = haar_dwt(x)
        energies = bands.square().mean(dim=(-2, -1))  # B×4×C
        weights = self.energy_mlp(energies).softmax(dim=1)
        frequency = haar_iwt(bands * weights[..., None, None], size)
        return x + self.fuse(torch.cat((self.spatial(x), frequency), dim=1))


class CoordinateAttention(nn.Module):
    """沿高宽分别聚合，生成 B×C×H×1 与 B×C×1×W 两个方向权重。"""

    def __init__(self, channels, reduction, minimum):
        super().__init__()
        hidden = max(minimum, channels//reduction)
        self.reduce = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False),
                                    nn.BatchNorm2d(hidden), nn.SiLU(inplace=True))
        self.height = nn.Conv2d(hidden, channels, 1)
        self.width = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        along_h = x.mean(dim=3, keepdim=True)
        along_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        h_feature, w_feature = torch.split(self.reduce(torch.cat((along_h, along_w), dim=2)), (h, w), dim=2)
        return x * self.height(h_feature).sigmoid() * self.width(w_feature.transpose(2, 3)).sigmoid()


class BoundaryGate(nn.Module):
    """每级重新分解对应跳连，不复用 SFF 子带；生成单通道空间边界门。"""

    def __init__(self, skip_channels, hidden):
        super().__init__()
        self.predict = nn.Sequential(nn.Conv2d(3*skip_channels, hidden, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(hidden), nn.SiLU(inplace=True),
                                     nn.Conv2d(hidden, 1, 1))

    def forward(self, decoded, skip):
        bands, _ = haar_dwt(skip)
        # 保留三个带符号的高频子带，避免自行假定高频只能来自裂隙。
        high = torch.cat(tuple(bands[:, k] for k in (1, 2, 3)), dim=1)
        gate = F.interpolate(self.predict(high), size=decoded.shape[-2:], mode="bilinear", align_corners=False)
        return decoded * gate.sigmoid() + decoded


class LAD(nn.Module):
    """双线性上采样、跳连融合、坐标注意力及残差边界门控。"""

    def __init__(self, incoming, skip_channels, outgoing, config):
        super().__init__()
        self.fuse = DSC(incoming+skip_channels, outgoing)
        self.attention = CoordinateAttention(outgoing, config["coordinate_reduction"], config["coordinate_min_channels"])
        self.boundary = BoundaryGate(skip_channels, config["boundary_channels"])

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.boundary(self.attention(self.fuse(torch.cat((x, skip), dim=1))), skip)
