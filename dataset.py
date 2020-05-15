import os
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class AdaINDataset(data.Dataset):
    def __init__(self, image_dir, opt):
        self.images = load_files(image_dir)
        self.transforms = transforms.Compose([
            transforms.RandomCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=opt.mean, std=opt.std)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        scaled_image = scale_image(Image.open(image_path).convert('RGB'))
        image_tensor = self.transforms(scaled_image)
        return image_tensor


def scale_image(image, shortside=512):
    w, h = image.size
    scale_ratio = shortside / min(w, h)
    sw, sh = int(w * scale_ratio), int(h * scale_ratio)
    scaled_image = image.resize((sw, sh), Image.BICUBIC)
    return scaled_image


def is_img_file(file_name):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tiff', '.webp']
    return any(file_name.endswith(extension) for extension in IMG_EXTENSIONS)


# 递归地返回文件夹下所有文件的路径（包括子文件夹下的文件）
# 只返回图像文件
def load_files(root_dir):
    assert os.path.exists(root_dir), f'"{root_dir}" does not exist'
    dir_valid = os.path.isdir(root_dir) or os.path.islink(root_dir)
    error_info = f'"{root_dir}" is not a valid directory'
    assert dir_valid, error_info

    file_paths = []
    for dir_path, dir_names, file_names in os.walk(root_dir):
        for file_name in file_names:
            if is_img_file(file_name):
                filepath = os.path.join(dir_path, file_name)
                file_paths.append(filepath)
    np.random.shuffle(file_paths)
    return file_paths
