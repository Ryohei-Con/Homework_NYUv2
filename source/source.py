import os
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as data
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision.datasets import VisionDataset
from torchvision import tv_tensors
import torchvision.transforms.v2 as v2
from torch.cuda.amp import autocast, GradScaler
from dataclasses import dataclass

from transformers import SegformerModel

from Depth2HHA_python_master.getHHA import wrap_getHHA
from SegFormer_block import Block, OverlapPatchMerging

# カラーマップ生成関数：セグメンテーションの可視化用
def colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


# NYUv2データセット：RGB画像、セグメンテーション、深度、法線マップを提供するデータセット
class NYUv2(Dataset):
    """NYUv2 dataset

    Args:
        root (string): Root directory path.
        split (string, optional): 'train' for training set, and 'test' for test set. Default: 'train'.
        target_type (string, optional): Type of target to use, ``semantic``, ``depth``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """
    cmap = colormap()
    def __init__(self,
                root: str,
                config,
                split='train',
                ):
        super().__init__()

        # データセットの基本設定
        assert(split in ('train', 'test'))
        self.root = root
        self.split = split
        self.train_idx = np.array([255, ] + list(range(13)))  # 13クラス分類用
        self.config = config

        # 画像ファイルのパスリストを作成
        img_names = os.listdir(os.path.join(self.root, self.split, 'image'))
        img_names.sort()
        images_dir = os.path.join(self.root, self.split, 'image')
        self.images = [os.path.join(images_dir, name) for name in img_names]

        label_dir = os.path.join(self.root, self.split, 'label')
        if (self.split == 'train'):
            self.targets = [os.path.join(label_dir, name) for name in img_names]

        depth_dir = os.path.join(self.root, self.split, 'depth')
        self.depths = [os.path.join(depth_dir, name) for name in img_names]

    @property
    def transform(self):
        return v2.Compose([
            v2.RandomHorizontalFlip(0.5),
            v2.RandomRotation((-10, 10)),
            v2.RandomResizedCrop(self.config.image_size),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @property
    def color_jitter(self):  # RGBデータのみに適用
        return v2.Compose([
            v2.ColorJitter()
        ])

    @property
    def test_transform(self):
        return v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, idx):
        if self.split == 'train':
            image = Image.open(self.images[idx])
            depth = Image.open(self.depths[idx])
            target = Image.open(self.targets[idx])

            image = v2.functional.pil_to_tensor(image).to(torch.float32) / 255.0
            depth = v2.functional.pil_to_tensor(depth).to(torch.float32) / 65535.0
            target = v2.functional.pil_to_tensor(target).to(torch.int32)

            # HHA encodingする
            depth = np.array(depth)
            hha_depth = wrap_getHHA(depth)

            # (B, C, H, W)の形でラップする
            image = tv_tensors.Image(torch.from_numpy(np.array(image)) / 255.0)
            depth = tv_tensors.Image(torch.from_numpy(hha_depth).permute(0, 3, 1, 2))
            target = tv_tensors.Mask(target)

            image, depth, target = self.transform(image, depth, target)
            image = self.color_jitter(image)

            return image, depth, target

        if self.split=='test':
            image = Image.open(self.images[idx])
            depth = Image.open(self.depths[idx])

            # HHA encodingする
            depth = np.array(depth)
            hha_depth = wrap_getHHA(depth)

            image = tv_tensors.Image(torch.from_numpy(np.array(image)) / 255.0)
            depth = tv_tensors.Image(torch.from_numpy(hha_depth).permute(0, 3, 1, 2))

            image = self.test_transform(image)
            depth = self.test_transform(depth)
            return image, depth

    def __len__(self):
        return len(self.images)


class DCMAF(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        """
        Discriminative Cross-Modal Attention Fusion (DCMAF) Module. 

        Args:
            num_channels (int): channels of feature map
        """
        super().__init__()
        self.rgb_process = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_channels, num_channels//2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(num_channels//2, num_channels, kernel_size=1),
        )
        self.d_process = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_channels, num_channels//2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(num_channels//2, num_channels, kernel_size=1),
        )

    def forward(self, rgb_map: torch.Tensor, depth_map: torch.Tensor):
        """

        Args:
            rgb_map (torch.Tensor): (B, C, H, W)
            depth_map (torch.Tensor): (B, C, H, W)

        Returns:
            torch.Tensor: (B, C, H, W)
        """
        print(f"{rgb_map.size()=}")
        print(f"{depth_map.size()=}")
        rgb_vec = self.rgb_process(rgb_map)
        depth_vec = self.d_process(depth_map)
        w = F.softmax(rgb_vec - depth_vec, dim=1)
        return w * rgb_map + (1-w) * depth_map


class MyEncoder(nn.Module):
    def __init__(self, RGB_model_name, D_model_name, dims, reductions, depths):
        super().__init__()
        self.RGB_model = SegformerModel.from_pretrained(
            RGB_model_name
        )
        self.D_model = SegformerModel.from_pretrained(
            D_model_name
        )
        self.RGB_backbone = self.RGB_model.encoder
        self.D_backbone = self.D_model.encoder

        # stage 2
        self.dcmaf2 = DCMAF(dims[1])
        self.blocks2 = nn.ModuleList([Block(dims[1], reduction=reductions[1]) for _ in range(depths[1])])
        self.patch_merge2 = OverlapPatchMerging(dims[1], dims[2], padding=1, stride=2, kernel=3)

        # stage 3
        self.dcmaf3 = DCMAF(dims[2])
        self.blocks3 = nn.ModuleList([Block(dims[2], reduction=reductions[2]) for _ in range(depths[2])])
        self.patch_merge3 = OverlapPatchMerging(dims[2], dims[3], padding=1, stride=2, kernel=3)

        # stage 4
        self.dcmaf4 = DCMAF(dims[3])
        self.blocks4 = nn.ModuleList([Block(dims[3], reduction=reductions[3]) for _ in range(depths[3])])
        self.patch_merge4 = OverlapPatchMerging(dims[3], dims[4], padding=1, stride=2, kernel=3)

        # stage 5
        # ここのpatch_mergeはfeature mapのサイズを変えないようにする
        print(f"{dims[4]=}")
        self.dcmaf5 = DCMAF(dims[4])
        self.blocks5 = nn.ModuleList([Block(dims[4], reduction=reductions[4]) for _ in range(depths[4])])

    def forward(self, RGB_image, D_image):
        outputs = []
        RGB_outputs = self.RGB_backbone(
            pixel_values=RGB_image,
            output_hidden_states=True
        )
        D_outputs = self.D_backbone(
            pixel_values=D_image,
            output_hidden_states=True
        )

        RGB_hidden_states = RGB_outputs.hidden_states
        D_hidden_states = D_outputs.hidden_states

        # stage 2
        hidden_state: torch.Tensor = self.dcmaf2(RGB_hidden_states[0], D_hidden_states[0])
        _, _, h, w = hidden_state.size()
        x = hidden_state.flatten(2).permute(0, 2, 1)
        for block in self.blocks2:
            x = block(x, h, w)
        b, l, dim = x.size()
        x = x.reshape(b, h, w, dim).permute(0, 3, 1, 2)
        outputs.append(x)
        print(f"in stage 2(before patch merge)\n{x.size()=}")
        x, h, w = self.patch_merge2(x)
        print(f"in stage 2(after patch merge)\n{x.size()=}\n{h=}\n{w=}")

        # stage 3
        hidden_state = self.dcmaf3(RGB_hidden_states[1], D_hidden_states[1])
        _, _, h, w = hidden_state.size()
        hidden_state = hidden_state.flatten(2).permute(0, 2, 1)
        print(f"{x.size()=}")
        print(f"{hidden_state.size()=}")
        x += hidden_state
        for block in self.blocks3:
            x = block(x, h, w)
        b, l, dim = x.size()
        x = x.reshape(b, h, w, dim).permute(0, 3, 1, 2)
        outputs.append(x)
        x, h, w = self.patch_merge3(x)

        # stage 4
        hidden_state = self.dcmaf4(RGB_hidden_states[2], D_hidden_states[2])
        _, _, h, w = hidden_state.size()
        hidden_state = hidden_state.flatten(2).permute(0, 2, 1)
        x += hidden_state
        for block in self.blocks4:
            x = block(x, h, w)
        b, l, dim = x.size()
        x = x.reshape(b, h, w, dim).permute(0, 3, 1, 2)
        outputs.append(x)
        x, h, w = self.patch_merge4(x)

        # stage 5
        print("stage 5")
        hidden_state = self.dcmaf5(RGB_hidden_states[3], D_hidden_states[3])
        _, _, h, w = hidden_state.size()
        hidden_state = hidden_state.flatten(2).permute(0, 2, 1)
        x += hidden_state
        for block in self.blocks5:
            x = block(x, h, w)
        b, l, dim = x.size()
        x = x.reshape(b, h, w, dim).permute(0, 3, 1, 2)
        outputs.append(x)
        return outputs
