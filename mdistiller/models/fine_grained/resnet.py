import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

# ImageNet 预训练权重 URL（PyTorch 1.7.0 官方链接）
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# ResNet 基础块（用于 ResNet18/34）
class BasicBlock(nn.Module):
    expansion = 1  # 输出通道数扩展倍数（BasicBlock 无扩展）

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample  # 下采样模块（解决残差连接维度不匹配）
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 若有下采样，对残差路径进行下采样（保证维度一致）
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# ResNet 瓶颈块（用于 ResNet50/101/152，含 1x1+3x3+1x1 卷积）
class Bottleneck(nn.Module):
    expansion = 4  # 输出通道数扩展倍数（1x1卷积将通道数放大4倍）

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet 主类
    Args:
        block (BasicBlock/Bottleneck): 基础块类型
        layers (list): 各层块数量（如 ResNet50 为 [3,4,6,3]）
        num_classes (int): 分类类别数（细粒度任务需指定）
        pretrained (bool/str): 预训练配置（True=自动下载，str=本地路径，False=随机初始化）
    """
    def __init__(self, block, layers, num_classes=1000, pretrained=False):
        super(ResNet, self).__init__()
        self.inplanes = 64  # 初始输入通道数
        self.pretrained = pretrained

        # 第一层：Conv + BN + ReLU（无池化，高分辨率输入需保留特征）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 下采样至 1/4 分辨率

        # 四层残差块（根据 layers 定义数量）
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 全局平均池化 + 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 适配任意输入分辨率，输出 (1,1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 分类头（动态替换）

        # 初始化参数（除预训练权重外的随机初始化）
        self._init_weights()

        # 加载预训练权重
        if self.pretrained:
            self._load_pretrained_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        """创建残差块层（堆叠 blocks 个基础块）"""
        downsample = None
        # 若 stride !=1 或输入通道数 != 输出通道数，需下采样残差路径
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # 第一个块可能含下采样
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        # 后续块无下采样
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self):
        """初始化未被预训练权重覆盖的参数"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _load_pretrained_weights(self):
        """加载 ImageNet 预训练权重（支持自动下载/本地路径）"""
        import os
        if isinstance(self.pretrained, str):
            # 本地权重路径
            if not os.path.exists(self.pretrained):
                raise FileNotFoundError(f"本地权重文件不存在：{self.pretrained}")
            print(f"加载本地 ResNet 预训练权重：{self.pretrained}")
            state_dict = torch.load(self.pretrained, map_location='cpu')
        else:
            # 自动下载权重（根据模型结构匹配 URL）
            model_name = self._get_model_name()
            if model_name not in model_urls:
                raise ValueError(f"不支持自动下载该 ResNet 变体：{model_name}")
            print(f"从官方 URL 下载 ResNet 预训练权重：{model_urls[model_name]}")
            state_dict = load_state_dict_from_url(model_urls[model_name], progress=True, map_location='cpu')

        # 替换分类头（预训练是 1000 类，当前是 num_classes 类，故删除原 fc 权重）
        if 'fc.weight' in state_dict and state_dict['fc.weight'].shape[0] != self.fc.out_features:
            del state_dict['fc.weight']
            del state_dict['fc.bias']
            print(f"删除预训练分类头权重（1000类 → {self.fc.out_features}类），分类头将重新初始化")

        # 加载权重（strict=False 忽略分类头不匹配）
        msg = self.load_state_dict(state_dict, strict=False)
        print(f"ResNet 预训练权重加载完成，未匹配键：{msg.missing_keys}")

    def _get_model_name(self):
        """根据层结构推断模型名称（如 ResNet50）"""
        block_type = self.layer1[0].__class__.__name__
        layer_nums = [len(self.layer1), len(self.layer2), len(self.layer3), len(self.layer4)]
        if block_type == 'BasicBlock':
            if layer_nums == [2, 2, 2, 2]:
                return 'resnet18'
            elif layer_nums == [3, 4, 6, 3]:
                return 'resnet34'
        elif block_type == 'Bottleneck':
            if layer_nums == [3, 4, 6, 3]:
                return 'resnet50'
            elif layer_nums == [3, 4, 23, 3]:
                return 'resnet101'
            elif layer_nums == [3, 8, 36, 3]:
                return 'resnet152'
        raise ValueError(f"无法识别 ResNet 变体：block={block_type}, layers={layer_nums}")

    def forward(self, x):
        """ResNet 前向传播（输出 logit）"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 展平为 (batch_size, 512*expansion)
        x = self.fc(x)  # 输出 logit：(batch_size, num_classes)

        return x


# 便捷函数：创建不同规格的 ResNet 模型
def resnet18(num_classes=1000, pretrained=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, pretrained=pretrained)


def resnet34(num_classes=1000, pretrained=False):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, pretrained=pretrained)


def resnet50(num_classes=1000, pretrained=False):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, pretrained=pretrained)


def resnet101(num_classes=1000, pretrained=False):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, pretrained=pretrained)


def resnet152(num_classes=1000, pretrained=False):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, pretrained=pretrained)


# 测试代码（可注释）
# if __name__ == "__main__":
#     # 测试 ResNet50 适配 CUB-200（200类）
#     model = resnet50(num_classes=200, pretrained=True)
#     x = torch.randn(2, 3, 448, 448)  # 2个样本，448x448分辨率
#     out = model(x)
#     print(f"ResNet50 输出形状：{out.shape}（应为 (2, 200)）")
#     print(f"分类头权重均值：{model.fc.weight.data.mean():.4f}（重新初始化后接近 0）")