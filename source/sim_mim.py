import datetime
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as v2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Subset
from tqdm import tqdm
from transformers import SegformerModel

from config import TrainingConfig, config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# mask image
class MaskImage(nn.Module):
    def __init__(self, image_size=(480, 640), patch_size=(40, 40), mask_ratio=0.5):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_num = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.patch_len = self.patch_num[0] * self.patch_num[1]
        self.mask = nn.Parameter(torch.randn((1, 1, 1, patch_size[0], patch_size[1])))  # (1, 1, 1, patch_H, patch_W)

    def forward(self, image: torch.Tensor):
        """
        マスクしてSSL用のインプット画像を作る
        また、損失関数用のmask_tensor(mask部分が1でそれ以外が0のテンソル)もつくる

        Args:
            image (torch.Tensor): (B, C, H, W)
        """
        b, c, h, w = image.size()
        remain_N = int(self.patch_len * (1 - self.mask_ratio))

        # パッチ化する -> (b, c, num_patches_H, num_patches_W, patch_H, patch_W)
        image_patch = image.reshape((b, c, self.patch_num[0], self.patch_size[0], self.patch_num[1], self.patch_size[1]))
        image_patch = image_patch.permute(0, 1, 2, 4, 3, 5).flatten(2, 3)

        # マスクする
        random = torch.rand((b, c, self.patch_num[0], self.patch_size[0], self.patch_num[1], self.patch_size[1]))
        masker = (random < self.mask_ratio).float()  # 1: masked, 0: visible
        masked_image = (1 - masker) * image + masker * self.mask
        masked_image = masked_image.permute(0, 1, 2, 4, 3, 5).reshape(b, c, h, w)
        return masked_image, masker


# make a model
class SimMiM(nn.Module):
    def __init__(self, encoder_model, dims, hid_channel, image_size):
        super().__init__()
        self.encoder = SegformerModel.from_pretrained(
            encoder_model
        )
        self.linear1 = nn.Conv2d(dims[1], hid_channel, 1)
        self.linear2 = nn.Conv2d(dims[2], hid_channel, 1)
        self.linear3 = nn.Conv2d(dims[3], hid_channel, 1)
        self.linear4 = nn.Conv2d(dims[4], hid_channel, 1)
        self.final_linear = nn.Conv2d(4 * hid_channel, 3, 1)
        self.upsample = nn.Upsample(image_size)

    def forward(self, masked_image: torch.Tensor):
        """encoder and decoder
        Args:
            masked_image (torch.Tensor): masked by MaskImage class. (B, C, H, W)
        """
        b, c, h, w = masked_image.size()
        outputs = self.encoder(masked_image, output_hidden_states=True).hidden_states
        f1 = self.linear1(outputs[0])
        f2 = self.linear2(outputs[1])
        f3 = self.linear3(outputs[2])
        f4 = self.linear4(outputs[3])
        pred = self.final_linear(torch.cat((f1, f2, f3, f4), dim=1))
        pred = self.upsample(pred)
        return pred


# calculate loss
def calc_loss(masked_image, pred, mask_tensor, num_masks):
    return (torch.abs(pred - masked_image) * mask_tensor).sum() / num_masks


# Dataset for Sim-MiM
# NYUv2データセット：RGB画像、セグメンテーション、深度、法線マップを提供するデータセット
class NYUv2(Dataset):
    """NYUv2 dataset

    Args:
        root (string): Root directory path.
        config (TrainingConfig): configuration of the dataset
        split (string, optional): 'train' for training set, and 'test' for test set. Default: 'train'.
    """
    def __init__(self, root: str, config, split='train'):
        super().__init__()

        # データセットの基本設定
        assert (split in ('train', 'test'))
        self.root = Path(root)
        self.split = split
        self.train_idx = np.array([255, ] + list(range(13)))  # 13クラス分類用
        self.config = config

        # 画像ファイルのパスリストを作成
        img_names = os.listdir(Path(self.root / self.split / 'image'))
        img_names.sort()

        hha_dir = Path(self.root / self.split / 'hha')
        self.hha = [Path(hha_dir / name) for name in img_names]

    @property
    def transform(self):
        return v2.Compose([
            v2.RandomHorizontalFlip(0.5),
            v2.RandomRotation((-10, 10)),
            v2.RandomResizedCrop(self.config.image_size),
        ])

    def __getitem__(self, idx):
        hha = Image.open(self.hha[idx])
        # (C, H, W)の形でラップする
        const = torch.Tensor([218.0, 255.0, 255.0])  # calc_const.pyで導いた。const_value.pklに保存
        hha = torch.from_numpy(hha).permute(2, 0, 1) / const
        hha = self.transform(hha)
        return hha

    def __len__(self):
        return len(self.hha)


# train, valid
class Trainer:
    def __init__(self, model: SimMiM, optimizer: optim.Adam, config: TrainingConfig, device: str):
        self.model = model.to(device)
        self.config = config
        self.optimizer = optimizer

    def train_one_epoch(self, dataloader: DataLoader, mode: str):
        if mode == "train":
            self.model.train()
        elif mode == "valid":
            self.model.eval()
        else:
            raise ValueError("mode must be train or valid")
        total_loss = 0
        mask_image = MaskImage(self.config.image_size, self.config.patch_size)
        for batch in tqdm(dataloader, desc="学習しています"):
            batch = batch.to(self.config.device)
            masked_batch, mask_tensor = mask_image(batch)
            num_masks = mask_tensor.sum()
            pred: torch.Tensor = self.model(masked_batch)
            loss = calc_loss(batch, pred, mask_tensor, num_masks)
            total_loss += loss

            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return total_loss

    def get_optimizer(self):
        # 将来的にはwarm upつきcosine annealingとかしたい
        pass

    def fit(self, dataset: NYUv2):
        early_finish_cnt = 0
        total_train_losses = []
        for epoch in range(self.config.num_epochs):
            total_train_loss = 0

            train_dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

            train_loss = self.train_one_epoch(train_dataloader, mode="train")
            total_train_loss += train_loss

            total_train_losses.append(total_train_loss)
            if total_train_losses[-1] > total_train_losses[-2] * 0.99:
                if early_finish_cnt == 0:
                    logger.info("""total_valid_loss didn't make a big improvement.
                                Finish fitting when there's almost no improvement in next 5 epochs""")
                    early_finish_cnt += 1
                elif early_finish_cnt < 5:
                    early_finish_cnt += 1
                else:
                    logger.info("finish fitting because there's almost no improvement in the last 5 epochs")
                    self.save_weights()
                    return None
            else:
                early_finish_cnt = 0
            logger.info(f"Epoch {epoch} / {self.config.num_epochs} has been finished.")
        logger.info("finished fitting")
        return None

    def save_weights(self):
        date = str(datetime.datetime.now())[:10]
        file_name = f"{date}-simmim_weights.pt"
        torch.save(self.model, file_name)
        logger.info(f"weights are saved at {file_name}")


if __name__ == "__main__":
    nyuv2_dataset = NYUv2(root="./data", config=config, split="train")
    model = SimMiM(
        config.depth_model,
        config.dims,
        config.decoder_hiddim,
        config.image_size
    )
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    trainer = Trainer(model, optimizer, config, "cpu")
    trainer.fit(dataset=nyuv2_dataset)
    logger.info("all processes are successfully finished")
