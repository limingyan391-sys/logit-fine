import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url


__all__ = ['MobileNetV2', 'mobilenet_v2']

# ImageNet 预训练权重 URL（PyTorch 1.7.0 官方链接）
model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


# MobileNetV2 核心模块：逆残差块（Inverted Residual）+ 线性瓶颈（Linear Bottleneck）
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2], "stride must be 1 or 2"

        hidden_dim = int(round(inp * expand_ratio))  # 扩展后的通道数（逆残差：先升维再降维）
        self.use_res_connect = self.stride == 1 and inp == oup  # 是否使用残差连接（stride=1且维度一致）

        layers = []
        # 1. 1x1 卷积（升维，激活函数用 ReLU6）
        if expand_ratio != 1:
            layers.append(nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        # 2. 3x3 深度可分离卷积（降维，激活函数用 ReLU6）
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))
        # 3. 1x1 卷积（线性瓶颈，无激活函数）
        layers.append(nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(oup))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """
    MobileNetV2 主类
    Args:
        num_classes (int): 分类类别数
        pretrained (bool/str): 预训练配置
        width_mult (float): 宽度系数（0 < width_mult ≤1，越小模型越轻量）
    """
    def __init__(self, num_classes=1000, pretrained=False, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        self.pretrained = pretrained
        self.width_mult = width_mult

        # MobileNetV2 原始通道数配置（按宽度系数缩放）
        input_channel = 32
        last_channel = 1280
        # 配置表：(expansion, out_channel, num_blocks, stride)
        inverted_residual_setting = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        # 1. 第一层：Conv + BN + ReLU6
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))  # 最后一层通道数缩放
        features = [
            nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        ]

        # 2. 逆残差块层（按配置表堆叠）
        for expansion, out_channel, num_blocks, stride in inverted_residual_setting:
            output_channel = int(out_channel * width_mult)
            for i in range(num_blocks):
                stride_block = stride if i == 0 else 1  # 仅第一个块用 stride（下采样）
                features.append(InvertedResidual(input_channel, output_channel, stride_block, expansion))
                input_channel = output_channel

        # 3. 最后一层：1x1 Conv + BN + ReLU6
        features.append(nn.Conv2d(input_channel, self.last_channel, kernel_size=1, stride=1, padding=0, bias=False))
        features.append(nn.BatchNorm2d(self.last_channel))
        features.append(nn.ReLU6(inplace=True))

        # 特征提取器（Sequential 包装）
        self.features = nn.Sequential(*features)

        # 4. 分类头（全局平均池化 + 线性层）
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.last_channel, num_classes)

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
        """加载预训练权重"""
        import os
        if isinstance(self.pretrained, str):
            if not os.path.exists(self.pretrained):
                raise FileNotFoundError(f"本地权重文件不存在：{self.pretrained}")
            state_dict = torch.load(self.pretrained, map_location='cpu')
            print(f"加载本地 MobileNetV2 预训练权重：{self.pretrained}")
        else:
            # 自动下载官方权重（仅 width_mult=1.0 支持）
            if self.width_mult != 1.0:
                raise ValueError(f"仅 width_mult=1.0 支持自动下载预训练权重，当前为 {self.width_mult}")
            print(f"从官方 URL 下载 MobileNetV2 预训练权重：{model_urls['mobilenet_v2']}")
            state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'], progress=True, map_location='cpu')

        # 替换分类头（删除原 classifier 权重）
        if 'classifier.1.weight' in state_dict and state_dict['classifier.1.weight'].shape[0] != self.classifier.out_features:
            del state_dict['classifier.1.weight']
            del state_dict['classifier.1.bias']
            print(f"删除预训练分类头权重（1000类 → {self.classifier.out_features}类），分类头重新初始化")

        # 加载权重（注意：官方权重分类头是 nn.Sequential(Linear+Dropout+Linear)，当前是单 Linear，需调整键名）
        adjusted_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('classifier.1'):
                # 官方权重分类头最后一层是 'classifier.1'，当前是 'classifier'
                new_k = k.replace('classifier.1', 'classifier')
                adjusted_state_dict[new_k] = v
            else:
                adjusted_state_dict[k] = v

        msg = self.load_state_dict(adjusted_state_dict, strict=False)
        print(f"MobileNetV2 预训练权重加载完成，未匹配键：{msg.missing_keys}")

    def forward(self, x):
        """前向传播（输出 logit）"""
        x = self.features(x)  # 特征提取：(batch_size, last_channel, H, W)
        x = self.avgpool(x)   # 全局平均池化：(batch_size, last_channel, 1, 1)
        x = torch.flatten(x, 1)  # 展平：(batch_size, last_channel)
        x = self.classifier(x)   # 输出 logit：(batch_size, num_classes)
        return x


# 便捷函数：创建 MobileNetV2 模型（默认 width_mult=1.0）
def mobilenet_v2(num_classes=1000, pretrained=False, width_mult=1.0):
    return MobileNetV2(num_classes=num_classes, pretrained=pretrained, width_mult=width_mult)


# 测试代码（可注释）
# if __name__ == "__main__":
#     # 测试 MobileNetV2 适配 Stanford Cars（196类），width_mult=0.75（更轻量）
#     model = mobilenet_v2(num_classes=196, pretrained=True, width_mult=1.0)
#     x = torch.randn(2, 3, 448, 448)
#     out = model(x)
#     print(f"MobileNetV2 输出形状：{out.shape}（应为 (2, 196)）")
#     print(f"分类头权重均值：{model.classifier.weight.data.mean():.4f}")