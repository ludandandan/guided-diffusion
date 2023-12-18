import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs. 对一个数据集，创建一个生成器，生成（images, kwargs）对

    Each images is an NCHW float tensor, and the kwargs dict contains zero or 图像是一个NCHW浮点张量，kwargs字典包含零或
    more keys, each of which map to a batched Tensor of their own. 多个键，每个键都映射到自己的批处理张量。
    The kwargs dict can be used for class labels, in which case the key is "y" kwargs字典可用于类标签，此时键为“ y”
    and the values are integer tensors of class labels. 值是整数张量的类标签。

    :param data_dir: a dataset directory. 数据集目录
    :param batch_size: the batch size of each returned pair. 每个返回对的批量大小
    :param image_size: the size to which images are resized. 图像调整大小的大小
    :param class_cond: if True, include a "y" key in returned dicts for class 如果为True，则在返回的字典中包含“ y”键以获取类标签。
                       label. If classes are not available and this is true, an 如果类不可用并且为真，则会引发异常。
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order. 如果为True，则以确定的顺序产生结果。
    :param random_crop: if True, randomly crop the images for augmentation. 如果为True，则随机裁剪图像以进行增强。
    :param random_flip: if True, randomly flip the images for augmentation. 如果为True，则随机翻转图像以进行增强。默认是不裁剪，默认翻转
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename, 假设类是文件名的第一部分，例如“ cat_1234.png”。
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files] #获取类别，class_names是一个列表，里面是所有文件的类别，没有去重
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}#sorted_classes是一个字典，key是类别，value是类别的序号（序号是按照类别名进行排序的）
        classes = [sorted_classes[x] for x in class_names]#classes是一个列表，里面是所有文件的类别序号，没有去重
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None, #类别，是数字形式的
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx] #获取图片路径
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")#读入图片转换为RGB格式

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1 #将图片转换为float32类型，并且归一化到[-1,1]之间

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64) #将类别转换为int64类型，并且放在字典里，key是y
        return np.transpose(arr, [2, 0, 1]), out_dict #这里是将通道C放在第一个维度，即HWC转换为CHW


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
