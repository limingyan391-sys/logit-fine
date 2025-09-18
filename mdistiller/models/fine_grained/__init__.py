from .resnet import ResNet, resnet18, resnet34, resnet50, resnet101, resnet152
from .mobilenet_v2 import MobileNetV2, mobilenet_v2
from .efficientnet_b0 import EfficientNetB0, efficientnet_b0
from .vit import ViT, vit_base_patch16_448

fine_grained_model_dict = {
    "ResNet18": resnet18,
    "ResNet34": resnet34,
    "ResNet50": resnet50,
    "ResNet101": resnet101,
    "MobileNetV2": mobilenet_v2,
    "EfficientNetB0": efficientnet_b0,
    "ViT": vit_base_patch16_448
}
