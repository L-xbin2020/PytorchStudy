# 图片分类案例

from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):
    """
    root_dir: 训练数据文件夹路径  如：'datasets/train'
    label: 目标路径 如：'ants'
    """
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir

        # 完整路径的拼接 如 datasets/train/ants
        self.image_file_path = os.path.join(root_dir, label_dir)

        # 全部图文件加入列表里
        self.image_list = os.listdir(self.image_file_path)
        pass

    def __getitem__(self, item):
        # 通过item获取每个图文件的文件名
        self.image_name = self.image_list[item]

        self.image_path = os.path.join(self.root_dir, self.label_dir, self.image_name)
        self.img = Image.open(self.image_path)

        return self.img, self.label_dir

        pass

    def __len__(self):
        return len(self.image_list)
        pass


root_dir = 'datasets/train'
label_dir1 = 'ants'
label_dir2 = 'bees'

ants_data = MyData(root_dir, label_dir1)
bees_data = MyData(root_dir, label_dir2)


train_data = ants_data + bees_data
image, label = ants_data[1]



from torchvision import transforms
dir(transforms)
print(dir(transforms))
print("owho")