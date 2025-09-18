import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url


__all__ = ['EfficientNetB0', 'efficientnet_b0']

# 第三方预训练权重 URL（适配 PyTorch 1.7.0，来自 pytorch-image-models）
model_urls = {
    'efficientnet_b0': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth',
}


# 辅助函数：计算卷积层输出通道数（按宽度系数缩放）
def round_channels(channels, width_mult=1.0, min_depth=8):
    if width_mult == 1.0:
        return channels
    channels = int(channels * width_mult)
    return max(min_depth, channels)


# 辅助函数：计算卷积核大小（按深度系数缩放）
def round_repeats(repeats, depth_mult=1.0):
    if depth_mult == 1.0:
        return repeats
    return int(math.ceil(depth_mult * repeats))


# EfficientNet 核心模块：MBConv（含 SE 注意力）
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=1, se_ratio=0.25):
        super(MBConv, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        mid_channels = in_channels * expand_ratio  # 扩展后的通道数

        # 1. 1x1 卷积（升维，仅当 expand_ratio>1 时存在）
        self.expand_conv = None
        if expand_ratio > 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.SiLU(inplace=True)  # EfficientNet 用 SiLU 激活（PyTorch 1.7.0 需自定义或用近似）
            )

        # 2. 3x3/5x5 深度可分离卷积
        padding = (kernel_size - 1) // 2  # 保证输入输出分辨率一致（stride=1时）
        self.depth_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, 
                      padding=padding, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True)
        )

        # 3. SE 注意力模块（Squeeze-and-Excitation）
        self.se_module = None
        if se_ratio is not None and 0 < se_ratio <= 1:
            se_channels = int(in_channels * se_ratio)
            self.se_module = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # Squeeze：全局平均池化
                nn.Conv2d(mid_channels, se_channels, kernel_size=1, stride=1, padding=0, bias=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(se_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=True),
                nn.Sigmoid()  # Excitation：生成通道权重
            )

        # 4. 1x1 卷积（降维，线性瓶颈）
        self.project_conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)  # 无激活函数
        )

        # 残差连接（仅当 stride=1 且输入输出通道一致时）
        self.use_residual = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        residual = x

        # 1. 升维卷积
        if self.expand_conv is not None:
            x = self.expand_conv(x)

        # 2. 深度可分离卷积
        x = self.depth_conv(x)

        # 3. SE 注意力
        if self.se_module is not None:
            se_weight = self.se_module(x)
            x = x * se_weight  # 通道加权

        # 4. 降维卷积
        x = self.project_conv(x)

        # 残差连接
        if self.use_residual:
            x += residual

        return x


# 自定义 SiLU 激活函数（PyTorch 1.7.0 无内置 nn.SiLU，用 nn.Hardswish 近似或自定义）
class SiLU(nn.Module):
    def __init__(self, inplace=True):
        super(SiLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


# 替换 nn.SiLU 为自定义实现（适配 PyTorch 1.7.0）
nn.SiLU = SiLU


class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 主类（基于原始论文配置）
    Args:
        num_classes (int): 分类类别数
        pretrained (bool/str): 预训练配置
        width_mult (float): 宽度系数（B0 默认 1.0）
        depth_mult (float): 深度系数（B0 默认 1.0）
    """
    def __init__(self, num_classes=1000, pretrained=False, width_mult=1.0, depth_mult=1.0):
        super(EfficientNetB0, self).__init__()
        self.pretrained = pretrained
        self.width_mult = width_mult
        self.depth_mult = depth_mult

        # EfficientNet-B0 原始配置表（exp_ratio, kernel, repeats, in_ch, out_ch, stride, se_ratio）
        config = [
            (1, 3, 1, 32, 16, 1, 0.25),
            (6, 3, 2, 16, 24, 2, 0.25),
            (6, 5, 2, 24, 40, 2, 0.25),
            (6, 3, 3, 40, 80, 2, 0.25),
            (6, 5, 3, 80, 112, 1, 0.25),
            (6, 5, 4, 112, 192, 2, 0.25),
            (6, 3, 1, 192, 320, 1, 0.25),
        ]

        # 1. 第一层：Conv + BN + SiLU
        in_channels = round_channels(32, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True)
        )

        # 2. MBConv 层（按配置表堆叠）
        features = []
        for exp_ratio, kernel, repeats, in_ch, out_ch, stride, se_ratio in config:
            in_ch = round_channels(in_ch, width_mult)
            out_ch = round_channels(out_ch, width_mult)
            repeats = round_repeats(repeats, depth_mult)
            for i in range(repeats):
                # 仅第一个块用 stride（下采样）
                block_stride = stride if i == 0 else 1
                features.append(MBConv(
                    in_channels=in_ch if i == 0 else out_ch,
                    out_channels=out_ch,
                    kernel_size=kernel,
                    stride=block_stride,
                    expand_ratio=exp_ratio,
                    se_ratio=se_ratio
                ))
        self.features = nn.Sequential(*features)

        # 3. 最后一层：Conv + BN + SiLU
        last_channels = round_channels(1280, width_mult)
        self.head_conv = nn.Sequential(
            nn.Conv2d(320, last_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(last_channels),
            nn.SiLU(inplace=True)
        )

        # 4. 分类头：全局平均池化 + 线性层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(last_channels, num_classes)

        # 初始化参数
        self._init_weights()

        # 加载预训练权重
        if self.pretrained:
            self._load_pretrained_weights()

    def _init_weights(self):
        """初始化参数"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _load_pretrained_weights(self):
        """加载预训练权重（第三方权重）"""
        import os
        if isinstance(self.pretrained, str):
            if not os.path.exists(self.pretrained):
                raise FileNotFoundError(f"本地权重文件不存在：{self.pretrained}")
            state_dict = torch.load(self.pretrained, map_location='cpu')
            print(f"加载本地 EfficientNet-B0 预训练权重：{self.pretrained}")
        else:
            # 自动下载第三方权重（仅 B0 支持）
            print(f"从第三方 URL 下载 EfficientNet-B0 预训练权重：{model_urls['efficientnet_b0']}")
            state_dict = load_state_dict_from_url(model_urls['efficientnet_b0'], progress=True, map_location='cpu')

        # 调整权重键名（第三方权重键可能含 "model." 前缀，需删除）
        adjusted_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_k = k[6:]  # 去掉 "model." 前缀
                adjusted_state_dict[new_k] = v
            else:
                adjusted_state_dict[k] = v

        # 替换分类头（删除原分类头权重）
        if 'classifier.weight' in adjusted_state_dict and adjusted_state_dict['classifier.weight'].shape[0] != self.classifier.out_features:
            del adjusted_state_dict['classifier.weight']
            del adjusted_state_dict['classifier.bias']
            print(f"删除预训练分类头权重（1000类 → {self.classifier.out_features}类），分类头重新初始化")

        # 加载权重
        msg = self.load_state_dict(adjusted_state_dict, strict=False)
        print(f"EfficientNet-B0 预训练权重加载完成，未匹配键：{msg.missing_keys}")

    def forward(self, x):
        """前向传播（输出 logit）"""
        x = self.stem(x)          #  stem 层：(batch_size, 32, H/2, W/2)
        x = self.features(x)      #  MBConv 层：(batch_size, 320, H/32, W/32)
        x = self.head_conv(x)     #  head_conv 层：(batch_size, 1280, H/32, W/32)
        x = self.avgpool(x)       #  全局平均池化：(batch_size, 1280, 1, 1)
        x = torch.flatten(x, 1)   #  展平：(batch_size, 1280)
        x = self.classifier(x)    #  输出 logit：(batch_size, num_classes)
        return x


# 便捷函数：创建 EfficientNet-B0 模型
def efficientnet_b0(num_classes=1000, pretrained=False):
    return EfficientNetB0(num_classes=num_classes, pretrained=pretrained, width_mult=1.0, depth_mult=1.0)


# 测试代码（可注释）
# if __name__ == "__main__":
#     # 测试 EfficientNet-B0 适配 FGVC Aircraft（100类）
#     model = efficientnet_b0(num_classes=100, pretrained=True)
#     x = torch.randn(2, 3, 448, 448)
#     out = model(x)
#     print(f"EfficientNet-B0 输出形状：{out.shape}（应为 (2, 100)）")
#     print(f"分类头权重均值：{model.classifier.weight.data.mean():.4f}")