import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
import math
from torchvision.models.utils import load_state_dict_from_url


__all__ = ['ViT', 'vit_base_patch16_448']


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=448, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=12, qkv_bias=True, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape  # B=batch_size, N=num_tokens, C=dim
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features=768, hidden_features=3072, out_features=None, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_hidden_dim=3072, qkv_bias=True, drop=0.1, attn_drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadAttention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(
            in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, drop=drop
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_size=448,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_hidden_dim=3072,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        pretrained=False,
        pretrained_model_name="vit_base_patch16_448"
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.pretrained_model_name = pretrained_model_name

        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # 2. Class Token + Positional Embedding
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 3. Transformer Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # 4. Classification Head
        self.head = nn.Linear(embed_dim, num_classes)

        # 5. Initialization
        self._init_weights()

        # 6. Load Pretrained Weights
        if self.pretrained:
            self._load_pretrained_weights()

    def _init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.class_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _load_pretrained_weights(self):
        import os
        # 第三方 ViT 预训练权重 URL（适配 PyTorch 1.7.0）
        pretrained_urls = {
            "vit_base_patch16_448": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_448-80ecf9dd.pth",
            "vit_base_patch16_224": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth"
        }

        # 加载权重文件
        if isinstance(self.pretrained, str):
            if not os.path.exists(self.pretrained):
                raise FileNotFoundError(f"本地权重文件不存在：{self.pretrained}")
            state_dict = torch.load(self.pretrained, map_location='cpu')
            print(f"加载本地 ViT 预训练权重：{self.pretrained}")
        else:
            if self.pretrained_model_name not in pretrained_urls:
                raise ValueError(f"不支持的预训练模型：{self.pretrained_model_name}")
            state_dict = load_state_dict_from_url(pretrained_urls[self.pretrained_model_name], progress=True, map_location='cpu')
            print(f"从第三方 URL 下载 ViT 预训练权重：{pretrained_urls[self.pretrained_model_name]}")

        # 调整权重键名和位置编码
        key_mapping = {"pos_embedding": "pos_embed", "cls_token": "class_token", "head.fc": "head"}
        adjusted_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            for old_k, new_k_val in key_mapping.items():
                if old_k in new_k:
                    new_k = new_k.replace(old_k, new_k_val)
                    break

            # 调整位置编码分辨率
            if new_k == "pos_embed":
                pretrained_img_size = int(math.sqrt(v.shape[1] - 1))
                current_img_size = int(math.sqrt(self.pos_embed.shape[1] - 1))
                if pretrained_img_size != current_img_size:
                    print(f"调整位置编码：{pretrained_img_size}x{pretrained_img_size} → {current_img_size}x{current_img_size}")
                    # 提取 Patch 位置编码并插值
                    pos_embed_patch = v[:, 1:, :].permute(0, 2, 1).reshape(1, v.shape[2], pretrained_img_size, pretrained_img_size)
                    pos_embed_patch = F.interpolate(pos_embed_patch, size=(current_img_size, current_img_size), mode="bilinear", align_corners=False)
                    pos_embed_patch = pos_embed_patch.reshape(1, v.shape[2], -1).permute(0, 2, 1)
                    # 拼接 Class Token 位置编码
                    pos_embed_cls = v[:, :1, :]
                    v = torch.cat([pos_embed_cls, pos_embed_patch], dim=1)

            adjusted_state_dict[new_k] = v

        # 加载权重（忽略分类头不匹配）
        msg = self.load_state_dict(adjusted_state_dict, strict=False)
        print(f"ViT 预训练权重加载完成，未匹配键：{msg.missing_keys}")

    def forward(self, x):
        B = x.shape[0]
        # Patch Embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        # 拼接 Class Token
        class_token = self.class_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([class_token, x], dim=1)  # (B, num_patches+1, embed_dim)
        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # Transformer Encoder
        for layer in self.encoder_layers:
            x = layer(x)
        # 归一化 + 提取 Class Token
        x = self.norm(x)[:, 0, :]  # (B, embed_dim)
        # 分类头输出
        x = self.head(x)  # (B, num_classes)
        return x


# 便捷函数：创建 ViT-Base 448x448 模型
def vit_base_patch16_448(num_classes=1000, pretrained=False):
    return ViT(
        img_size=448,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_hidden_dim=3072,
        num_classes=num_classes,
        pretrained=pretrained,
        pretrained_model_name="vit_base_patch16_448"
    )


# 测试代码（可注释）
# if __name__ == "__main__":
#     model = vit_base_patch16_448(num_classes=200, pretrained=True)
#     x = torch.randn(2, 3, 448, 448)
#     out = model(x)
#     print(f"ViT 输出形状：{out.shape}（应为 (2, 200)）")
#     print(f"分类头权重均值：{model.head.weight.data.mean():.4f}")