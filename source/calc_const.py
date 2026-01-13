import os
import pickle
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms.v2 as v2
from config import TrainingConfig, config
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from Depth2HHA_python_master.getHHA import wrap_getHHA


# Data Loader
# NYUv2データセット：RGB画像、セグメンテーション、深度、法線マップを提供するデータセット
class NYUv2(Dataset):
    """NYUv2 dataset

    Args:
        root (string): Root data directory path.
        split (string, optional): 'train' for training set, and 'test' for test set. Default: 'train'.
        target_type (string, optional): Type of target to use, ``semantic``, ``depth``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """
    def __init__(self, root: str, config: TrainingConfig, split='train'):
        super().__init__()

        # データセットの基本設定
        assert (split in ('train', 'test'))
        self.root = Path(root)
        self.split = Path(split)
        self.train_idx = np.array([255, ] + list(range(13)))  # 13クラス分類用
        self.config = config

        # 画像ファイルのパスリストを作成
        img_names = os.listdir(os.path.join(self.root, self.split, 'image'))
        img_names.sort()

        depth_dir = Path(self.root / self.split / 'depth')
        self.depths = [Path(depth_dir / name) for name in img_names]

    def __getitem__(self, idx):
        depth_filename = self.depths[idx]
        file_name = depth_filename.name
        depth = Image.open(depth_filename)

        to_meter_c = 1000.0  # 単位をメートルに変換
        depth = v2.functional.pil_to_tensor(depth).to(torch.float32) / to_meter_c

        # HHA encodingする
        depth = np.array(depth).squeeze()
        hha_depth = wrap_getHHA(depth)
        pil_img = Image.fromarray(hha_depth)
        hha_filename = Path(f"./data/train/hha/{file_name}")
        pil_img.save(hha_filename)

        return hha_depth

    def __len__(self):
        return len(self.depths)


dataset = NYUv2("data", config, split="train")

HEIGHT = 480
WIDTH = 640
CHANNEL = 3
all_data_max = torch.zeros((3))
for data in tqdm(dataset):
    data = torch.Tensor(data)
    data_max = torch.squeeze(data.reshape((HEIGHT*WIDTH, CHANNEL)).max(dim=0).values)
    all_data_max = torch.maximum(all_data_max, data_max)

print(all_data_max)

with open("./const_value.pkl", "wb") as f:
    print("const_value is saved")
    pickle.dump(all_data_max, f)
