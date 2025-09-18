import os
import numpy as np
import pandas as pd
import scipy
from typing import Optional, Callable, Tuple, Sequence, Union
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import (
    download_url,
    extract_archive,
    download_file_from_google_drive,
    verify_str_arg
)
from torch.utils.data import Dataset


class CUB(VisionDataset):
    """`CUB-200-2011 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>`_ Dataset.
    
    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
           creates from test set. Default: True.
        transform (callable, optional): A function/transform that takes in an PIL image
           and returns a transformed version. E.g, ``transforms.RandomCrop``. Default: None.
        target_transform (callable, optional): A function/transform that takes in the
           target and transforms it. Default: None.
        download (bool, optional): If True, downloads the dataset from Google Drive and
           puts it in root directory. If dataset is already downloaded, it is not
           downloaded again. Default: False.
    """
    base_folder = 'CUB_200_2011/images'
    file_id = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.loader = default_loader
        self.train = train

        if download:
            self._download()
        
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it.')
        
        self._load_metadata()

    def _load_metadata(self) -> None:
        """加载数据集元信息（图像路径、类别标签、训练/测试划分）"""
        # 读取图像路径文件
        images_path = os.path.join(self.root, 'CUB_200_2011', 'images.txt')
        images = pd.read_csv(images_path, sep=' ', names=['img_id', 'filepath'])
        
        # 读取图像-类别映射
        label_path = os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt')
        image_class_labels = pd.read_csv(label_path, sep=' ', names=['img_id', 'target'])
        
        # 读取训练/测试划分
        split_path = os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt')
        train_test_split = pd.read_csv(split_path, sep=' ', names=['img_id', 'is_training_img'])
        
        # 合并数据
        self.data = images.merge(image_class_labels, on='img_id')
        self.data = self.data.merge(train_test_split, on='img_id')
        
        # 读取类别名称
        class_names_path = os.path.join(self.root, 'CUB_200_2011', 'classes.txt')
        class_names = pd.read_csv(class_names_path, sep=' ', names=['class_idx', 'class_name'], usecols=[1])
        self.class_names = class_names['class_name'].tolist()
        
        # 筛选训练集/测试集
        if self.train:
            self.data = self.data[self.data['is_training_img'] == 1]
        else:
            self.data = self.data[self.data['is_training_img'] == 0]

    def _check_integrity(self) -> bool:
        """检查数据集完整性（元信息是否存在、图像文件是否齐全）"""
        try:
            self._load_metadata()
        except Exception:
            return False
        
        # 检查所有图像文件是否存在
        for _, row in self.data.iterrows():
            img_path = os.path.join(self.root, self.base_folder, row['filepath'])
            if not os.path.isfile(img_path):
                print(f"Missing image file: {img_path}")
                return False
        return True

    def _download(self) -> None:
        """从Google Drive下载并解压数据集"""
        if self._check_integrity():
            print('Files already downloaded and verified.')
            return
        
        # 下载压缩包
        download_file_from_google_drive(
            file_id=self.file_id,
            root=self.root,
            filename=self.filename,
            md5=self.tgz_md5
        )
        
        # 解压
        tar_path = os.path.join(self.root, self.filename)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=self.root)
        
        # 可选：删除压缩包节省空间
        os.remove(tar_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        """获取指定索引的样本（图像+标签），标签从1-based转为0-based"""
        sample = self.data.iloc[idx]
        img_path = os.path.join(self.root, self.base_folder, sample['filepath'])
        target = sample['target'] - 1  # 转为0-based标签
        
        # 加载图像
        img = self.loader(img_path)
        
        # 应用变换
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target


class Cars(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset.
    
    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
           creates from test set. Default: True.
        transform (callable, optional): A function/transform that takes in an PIL image
           and returns a transformed version. E.g, ``transforms.RandomCrop``. Default: None.
        target_transform (callable, optional): A function/transform that takes in the
           target and transforms it. Default: None.
        download (bool, optional): If True, downloads the dataset from Stanford server and
           puts it in root directory. If dataset is already downloaded, it is not
           downloaded again. Default: False.
    """
    file_list = {
        'imgs': ('http://imagenet.stanford.edu/internal/car196/car_ims.tgz', 'car_ims.tgz'),
        'annos': ('http://imagenet.stanford.edu/internal/car196/cars_annos.mat', 'cars_annos.mat')
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.loader = default_loader
        self.train = train

        if download:
            self._download()
        
        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it.')
        
        # 加载标注数据并生成样本列表
        self._load_annotations()

    def _load_annotations(self) -> None:
        """加载MAT格式的标注文件，生成（图像路径，标签）列表"""
        anno_path = os.path.join(self.root, self.file_list['annos'][1])
        loaded_mat = scipy.io.loadmat(anno_path)['annotations'][0]  # 提取标注数组
        
        self.samples = []
        for item in loaded_mat:
            # 筛选训练集/测试集（MAT中最后一列1表示测试集，0表示训练集）
            is_test = bool(item[-1][0])
            if self.train == is_test:
                continue
            
            # 提取图像路径和标签（标签从1-based转为0-based）
            img_rel_path = str(item[0][0])  # 相对路径："car_ims/xxx.jpg"
            target = int(item[-2][0]) - 1
            self.samples.append((img_rel_path, target))

    def _check_exists(self) -> bool:
        """检查数据集是否存在（解压后的图像文件夹+标注文件）"""
        img_dir = os.path.join(self.root, 'car_ims')  # 解压后的图像目录
        anno_file = os.path.join(self.root, self.file_list['annos'][1])  # 标注文件
        return os.path.isdir(img_dir) and os.path.isfile(anno_file)

    def _download(self) -> None:
        """下载并解压图像和标注文件"""
        if self._check_exists():
            print('Files already downloaded and verified.')
            return
        
        # 下载图像压缩包和标注文件
        for url, filename in self.file_list.values():
            download_url(url, root=self.root, filename=filename)
        
        # 解压图像压缩包
        img_tar_path = os.path.join(self.root, self.file_list['imgs'][1])
        print(f'Extracting image archive: {img_tar_path}')
        extract_archive(img_tar_path, root=self.root)
        
        # 可选：删除压缩包节省空间
        os.remove(img_tar_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        """获取指定索引的样本（图像+标签）"""
        img_rel_path, target = self.samples[idx]
        img_path = os.path.join(self.root, img_rel_path)
        
        # 加载图像
        img = self.loader(img_path)
        
        # 应用变换
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target


class Dogs(VisionDataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.
    
    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
           creates from test set. Default: True.
        transform (callable, optional): A function/transform that takes in an PIL image
           and returns a transformed version. E.g, ``transforms.RandomCrop``. Default: None.
        target_transform (callable, optional): A function/transform that takes in the
           target and transforms it. Default: None.
        download (bool, optional): If True, downloads the dataset from Stanford server and
           puts it in root directory. If dataset is already downloaded, it is not
           downloaded again. Default: False.
    """
    download_url_prefix = 'http://vision.stanford.edu/aditya86/ImageNetDogs'

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.loader = default_loader
        self.train = train

        if download:
            self._download()
        
        # 加载训练/测试划分
        self.split = self._load_split()
        
        # 初始化数据集目录和样本列表
        self.images_folder = os.path.join(self.root, 'Images')
        self.annotations_folder = os.path.join(self.root, 'Annotation')
        self._breeds = list_dir(self.images_folder)  # 所有犬种名称
        self._flat_breed_images = [(anno + '.jpg', idx) for anno, idx in self.split]

    def _load_split(self) -> Sequence[Tuple[str, int]]:
        """加载训练/测试划分文件（MAT格式），返回（标注名称，标签）列表"""
        split_filename = 'train_list.mat' if self.train else 'test_list.mat'
        split_path = os.path.join(self.root, split_filename)
        
        # 读取MAT文件
        mat_data = scipy.io.loadmat(split_path)
        annotations = [item[0][0] for item in mat_data['annotation_list']]  # 标注名称（无后缀）
        labels = [item[0] - 1 for item in mat_data['labels']]  # 标签转为0-based
        
        return list(zip(annotations, labels))

    def _download(self) -> None:
        """下载并解压数据集（图像、标注、划分文件）"""
        # 检查是否已存在完整数据集
        required_dirs = [os.path.join(self.root, 'Images'), os.path.join(self.root, 'Annotation')]
        if all(os.path.isdir(dir_) and len(os.listdir(dir_)) == 120 for dir_ in required_dirs):
            print('Files already downloaded and verified.')
            return
        
        # 下载三个关键压缩包
        for filename in ['images', 'annotation', 'lists']:
            tar_filename = f'{filename}.tar'
            url = f'{self.download_url_prefix}/{tar_filename}'
            tar_path = os.path.join(self.root, tar_filename)
            
            # 下载
            download_url(url, root=self.root, filename=tar_filename)
            
            # 解压
            print(f'Extracting: {tar_path}')
            with tarfile.open(tar_path, 'r') as tar_file:
                tar_file.extractall(self.root)
            
            # 删除压缩包节省空间
            os.remove(tar_path)

    def stats(self) -> dict:
        """统计数据集信息（每类样本数）"""
        counts = {}
        for _, target in self._flat_breed_images:
            counts[target] = counts.get(target, 0) + 1
        
        total_samples = len(self._flat_breed_images)
        total_classes = len(counts)
        print(f"{total_samples} samples spanning {total_classes} classes (avg {total_samples/total_classes:.2f} per class)")
        return counts

    def __len__(self) -> int:
        return len(self._flat_breed_images)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        """获取指定索引的样本（图像+标签）"""
        img_anno_name, target = self._flat_breed_images[idx]
        img_path = os.path.join(self.images_folder, img_anno_name)
        
        # 加载图像
        img = self.loader(img_path)
        
        # 应用变换
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target


class Aircraft(Dataset):
    """`FGVC Aircraft <https://github.com/fgvc-aircraft/aircraft>`_ Dataset.
    注：默认使用"variant"类别划分（100类），支持trainval/test划分。
    
    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from trainval set, otherwise
           creates from test set. Default: True.
        transform (callable, optional): A function/transform that takes in an PIL image
           and returns a transformed version. E.g, ``transforms.RandomCrop``. Default: None.
    """
    img_folder = os.path.join('fgvc-aircraft-2013b', 'data', 'images')  # 相对根目录的图像路径
    class_type = 'variant'  # 类别划分类型：variant(100类)/family(7类)/manufacturer(41类)

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.split = 'trainval' if self.train else 'test'
        
        # 类别文件路径（按class_type和split划分）
        self.classes_file = os.path.join(
            self.root,
            'fgvc-aircraft-2013b',
            'data',
            f'images_{self.class_type}_{self.split}.txt'
        )
        
        # 加载类别映射和样本列表
        self.image_ids, self.targets, self.classes, self.class_to_idx = self._find_classes()
        self.samples = self._make_dataset()
        self.loader = default_loader

    def _find_classes(self) -> Tuple[list, list, list, dict]:
        """读取类别文件，返回（图像ID列表，标签列表，类别名称列表，类别-索引映射）"""
        image_ids = []
        target_names = []
        
        # 读取类别文件（每行格式："图像ID 类别名称"）
        with open(self.classes_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                split_line = line.split(' ', 1)  # 按第一个空格分割（图像ID与类别名称）
                image_ids.append(split_line[0])
                target_names.append(split_line[1])
        
        # 生成类别列表和类别-索引映射
        classes = sorted(np.unique(target_names))
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        # 转换标签为索引
        targets = [class_to_idx[cls] for cls in target_names]
        
        return image_ids, targets, classes, class_to_idx

    def _make_dataset(self) -> list:
        """生成样本列表（图像绝对路径，标签）"""
        assert len(self.image_ids) == len(self.targets), "Image IDs and targets length mismatch."
        
        samples = []
        for img_id, target in zip(self.image_ids, self.targets):
            img_filename = f'{img_id}.jpg'
            img_path = os.path.join(self.root, self.img_folder, img_filename)
            
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
            
            samples.append((img_path, target))
        
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        """获取指定索引的样本（图像+标签）"""
        img_path, target = self.samples[idx]
        
        # 加载图像
        img = self.loader(img_path)
        
        # 应用变换
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target


# ------------------------------ 测试代码（可选）------------------------------
# if __name__ == '__main__':
#     import torchvision.transforms as transforms

#     # 基础变换（用于测试）
#     test_transform = transforms.Compose([
#         transforms.Resize((448, 448)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     # 测试CUB数据集（需确保root路径正确，首次运行可设download=True）
#     print("=== Testing CUB Dataset ===")
#     cub_train = CUB(root='./data/cub', train=True, transform=test_transform, download=False)
#     print(f"CUB Train Set: {len(cub_train)} samples, {len(cub_train.class_names)} classes")
#     img, target = cub_train[0]
#     print(f"CUB Sample: img shape {img.shape}, target {target} (class: {cub_train.class_names[target]})\n")

#     # 测试Cars数据集
#     print("=== Testing Cars Dataset ===")
#     cars_train = Cars(root='./data/cars', train=True, transform=test_transform, download=False)
#     print(f"Cars Train Set: {len(cars_train)} samples, 196 classes")
#     img, target = cars_train[0]
#     print(f"Cars Sample: img shape {img.shape}, target {target}\n")

#     # 测试Dogs数据集
#     print("=== Testing Dogs Dataset ===")
#     dogs_train = Dogs(root='./data/dogs', train=True, transform=test_transform, download=False)
#     dogs_train.stats()
#     img, target = dogs_train[0]
#     print(f"Dogs Sample: img shape {img.shape}, target {target}\n")

#     # 测试Aircraft数据集
#     print("=== Testing Aircraft Dataset ===")
#     aircraft_train = Aircraft(root='./data/aircraft', train=True, transform=test_transform)
#     print(f"Aircraft Train Set: {len(aircraft_train)} samples, {len(aircraft_train.classes)} classes")
#     img, target = aircraft_train[0]
#     print(f"Aircraft Sample: img shape {img.shape}, target {target} (class: {aircraft_train.classes[target]})")