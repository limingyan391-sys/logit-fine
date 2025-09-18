from .cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample, get_cifar100_dataloaders_trainval, get_cifar100_dataloaders_val_only, get_cifar100_dataloaders_train_only, get_cifar100_dataloaders_strong
from .imagenet import get_imagenet_dataloaders, get_imagenet_dataloaders_sample, get_imagenet_dataloaders_strong
from .fine_grained import CUB, Dogs, Cars, Aircraft
from torch.utils.data import DataLoader  # 新增：用于构建细粒度数据集的dataloader
from torch.utils.data import Dataset


__all__ = [
    # ... 原有数据集
    "CUB200", "StanfordDogs", "StanfordCars", "FGVCAircraft"
]
# DATASET_REGISTRY = {}
# def register_dataset(name):
#     """装饰器：注册数据集类到DATASET_REGISTRY"""
#     def register_dataset_cls(cls):
#         if name in DATASET_REGISTRY:
#             raise ValueError(f"Dataset {name} already registered!")
#         if not issubclass(cls, Dataset):
#             raise ValueError(f"Dataset {cls.__name__} must inherit from torch.utils.data.Dataset")
#         DATASET_REGISTRY[name] = cls
#         return cls
#     return register_dataset_cls

# class BaseDataset(Dataset):
#     """基础数据集类（原有代码）"""
#     def __init__(self, root, split, transform=None):
#         self.root = root
#         self.split = split
#         self.transform = transform
#         self.samples = []  # 需在子类中初始化（图像路径+标签列表）
    
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, idx):
#         path, label = self.samples[idx]
#         img = Image.open(path).convert("RGB")
#         if self.transform:
#             img = self.transform(img)
#         return img, label
def get_dataset(cfg):
    # -------------------------- 原有：CIFAR100 支持 --------------------------
    if cfg.DATASET.TYPE == "cifar100":
        if cfg.DISTILLER.TYPE == "CRD":
            train_loader, val_loader, num_data = get_cifar100_dataloaders_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
                mode=cfg.CRD.MODE,
            )
        else:
            train_loader, val_loader, num_data = get_cifar100_dataloaders(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
            )
        num_classes = 100

    # -------------------------- 原有：ImageNet 支持 --------------------------
    elif cfg.DATASET.TYPE == "imagenet":
        if cfg.DISTILLER.TYPE == "CRD":
            train_loader, val_loader, num_data = get_imagenet_dataloaders_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
            )
        else:
            train_loader, val_loader, num_data = get_imagenet_dataloaders(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
            )
        num_classes = 1000

    # -------------------------- 新增：细粒度数据集支持 --------------------------
    # 1. CUB-200 数据集
    elif cfg.DATASET.TYPE == "CUB":
        # 构建训练集（train=True）
        train_dataset = CUB(
            root=cfg.DATASET.ROOT,  # 配置中的数据集根路径
            train=True,
            transform=cfg.DATASET.TRAIN_TRANSFORMS,  # 训练数据增强（从配置读取）
            download=cfg.DATASET.DOWNLOAD  # 可选：配置中添加是否自动下载
        )
        # 构建验证集（train=False）
        val_dataset = CUB(
            root=cfg.DATASET.ROOT,
            train=False,
            transform=cfg.DATASET.VAL_TRANSFORMS,  # 验证数据增强（从配置读取）
            download=cfg.DATASET.DOWNLOAD
        )
        # 包装为 DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=True,  # 训练集打乱
            num_workers=cfg.DATASET.NUM_WORKERS,
            pin_memory=True  # 加速GPU数据传输
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            shuffle=False,  # 验证集不打乱
            num_workers=cfg.DATASET.NUM_WORKERS,
            pin_memory=True
        )
        num_data = len(train_dataset)  # 训练集总数据量
        num_classes = 200  # CUB-200 固定200类

    # 2. Stanford Cars 数据集（196类）
    elif cfg.DATASET.TYPE == "Cars":
        train_dataset = Cars(
            root=cfg.DATASET.ROOT,
            train=True,
            transform=cfg.DATASET.TRAIN_TRANSFORMS,
            download=cfg.DATASET.DOWNLOAD
        )
        val_dataset = Cars(
            root=cfg.DATASET.ROOT,
            train=False,
            transform=cfg.DATASET.VAL_TRANSFORMS,
            download=cfg.DATASET.DOWNLOAD
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.DATASET.NUM_WORKERS,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.DATASET.NUM_WORKERS,
            pin_memory=True
        )
        num_data = len(train_dataset)
        num_classes = 196  # Stanford Cars 固定196类

    # 3. Stanford Dogs 数据集（120类）
    elif cfg.DATASET.TYPE == "Dogs":
        train_dataset = Dogs(
            root=cfg.DATASET.ROOT,
            train=True,
            transform=cfg.DATASET.TRAIN_TRANSFORMS,
            download=cfg.DATASET.DOWNLOAD
        )
        val_dataset = Dogs(
            root=cfg.DATASET.ROOT,
            train=False,
            transform=cfg.DATASET.VAL_TRANSFORMS,
            download=cfg.DATASET.DOWNLOAD
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.DATASET.NUM_WORKERS,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.DATASET.NUM_WORKERS,
            pin_memory=True
        )
        num_data = len(train_dataset)
        num_classes = 120  # Stanford Dogs 固定120类

    # 4. FGVC Aircraft 数据集（variant类别：100类）
    elif cfg.DATASET.TYPE == "Aircraft":
        train_dataset = Aircraft(
            root=cfg.DATASET.ROOT,
            train=True,
            transform=cfg.DATASET.TRAIN_TRANSFORMS
        )
        val_dataset = Aircraft(
            root=cfg.DATASET.ROOT,
            train=False,
            transform=cfg.DATASET.VAL_TRANSFORMS
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.DATASET.NUM_WORKERS,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.DATASET.NUM_WORKERS,
            pin_memory=True
        )
        num_data = len(train_dataset)
        num_classes = 100  # FGVC Aircraft（variant）固定100类

    # -------------------------- 未支持的数据集 --------------------------
    else:
        raise NotImplementedError(f"Dataset type {cfg.DATASET.TYPE} is not supported yet!")

    return train_loader, val_loader, num_data, num_classes


def get_dataset_strong(cfg):
    # 该函数用于「强数据增强」场景（如部分蒸馏方法需要），逻辑与get_dataset一致，仅替换训练transform为强增强版本
    # -------------------------- 原有：CIFAR100 强增强支持 --------------------------
    if cfg.DATASET.TYPE == "cifar100":
        if cfg.DISTILLER.TYPE == "CRD":
            train_loader, val_loader, num_data = get_cifar100_dataloaders_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
                mode=cfg.CRD.MODE,
            )
        else:
            train_loader, val_loader, num_data = get_cifar100_dataloaders_strong(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
            )
        num_classes = 100

    # -------------------------- 原有：ImageNet 强增强支持 --------------------------
    elif cfg.DATASET.TYPE == "imagenet":
        if cfg.DISTILLER.TYPE == "CRD":
            train_loader, val_loader, num_data = get_imagenet_dataloaders_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
            )
        else:
            train_loader, val_loader, num_data = get_imagenet_dataloaders_strong(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
            )
        num_classes = 1000

    # -------------------------- 新增：细粒度数据集强增强支持 --------------------------
    # 核心差异：训练集用 cfg.DATASET.TRAIN_TRANSFORMS_STRONG（强增强），验证集不变
    elif cfg.DATASET.TYPE == "CUB":
        train_dataset = CUB(
            root=cfg.DATASET.ROOT,
            train=True,
            transform=cfg.DATASET.TRAIN_TRANSFORMS_STRONG,  # 强数据增强
            download=cfg.DATASET.DOWNLOAD
        )
        val_dataset = CUB(
            root=cfg.DATASET.ROOT,
            train=False,
            transform=cfg.DATASET.VAL_TRANSFORMS,
            download=cfg.DATASET.DOWNLOAD
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.DATASET.NUM_WORKERS,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.DATASET.NUM_WORKERS,
            pin_memory=True
        )
        num_data = len(train_dataset)
        num_classes = 200

    elif cfg.DATASET.TYPE == "Cars":
        train_dataset = Cars(
            root=cfg.DATASET.ROOT,
            train=True,
            transform=cfg.DATASET.TRAIN_TRANSFORMS_STRONG,  # 强增强
            download=cfg.DATASET.DOWNLOAD
        )
        val_dataset = Cars(
            root=cfg.DATASET.ROOT,
            train=False,
            transform=cfg.DATASET.VAL_TRANSFORMS,
            download=cfg.DATASET.DOWNLOAD
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.DATASET.NUM_WORKERS,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.DATASET.NUM_WORKERS,
            pin_memory=True
        )
        num_data = len(train_dataset)
        num_classes = 196

    elif cfg.DATASET.TYPE == "Dogs":
        train_dataset = Dogs(
            root=cfg.DATASET.ROOT,
            train=True,
            transform=cfg.DATASET.TRAIN_TRANSFORMS_STRONG,  # 强增强
            download=cfg.DATASET.DOWNLOAD
        )
        val_dataset = Dogs(
            root=cfg.DATASET.ROOT,
            train=False,
            transform=cfg.DATASET.VAL_TRANSFORMS,
            download=cfg.DATASET.DOWNLOAD
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.DATASET.NUM_WORKERS,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.DATASET.NUM_WORKERS,
            pin_memory=True
        )
        num_data = len(train_dataset)
        num_classes = 120

    elif cfg.DATASET.TYPE == "Aircraft":
        train_dataset = Aircraft(
            root=cfg.DATASET.ROOT,
            train=True,
            transform=cfg.DATASET.TRAIN_TRANSFORMS_STRONG,  # 强增强
        )
        val_dataset = Aircraft(
            root=cfg.DATASET.ROOT,
            train=False,
            transform=cfg.DATASET.VAL_TRANSFORMS,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.DATASET.NUM_WORKERS,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.DATASET.NUM_WORKERS,
            pin_memory=True
        )
        num_data = len(train_dataset)
        num_classes = 100

    # -------------------------- 未支持的数据集 --------------------------
    else:
        raise NotImplementedError(f"Dataset type {cfg.DATASET.TYPE} is not supported in get_dataset_strong!")

    return train_loader, val_loader, num_data, num_classes

# register_dataset("cub", cub)
# register_dataset("cars", cars)
# register_dataset("dogs", dogs)
# register_dataset("aircraft", aircraft)