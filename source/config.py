import torch
from dataclasses import dataclass


# config
@dataclass
class TrainingConfig:
    # データセットパス
    dataset_root: str = "data"

    # データ関連
    batch_size: int = 8
    num_workers: int = 4

    # モデル関連
    in_channels: int = 3
    num_classes: int = 13  # NYUv2データセットの場合

    # 学習関連
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4

    # データ分割関連
    train_val_split: float = 0.8  # 訓練データの割合

    # デバイス設定
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # チェックポイント関連
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 5  # エポックごとのモデル保存間隔

    # データ拡張・前処理関連
    image_size: tuple = (480, 640)
    patch_size: tuple = (40, 40)
    normalize_mean: tuple = (0.485, 0.456, 0.406)  # ImageNetの標準化パラメータ
    normalize_std: tuple = (0.229, 0.224, 0.225)

    # 学習のハイパーパラメータ
    rgb_model: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    depth_model: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    dims: tuple = (3, 32, 64, 160, 256, 256)
    reductions: tuple = (64, 16, 4, 1, 1)
    depths: tuple = (3, 4, 18, 18, 3)  # mit_b3を参考にした
    decoder_hiddim: int = 768  # mit_b3を参考にした

    def __post_init__(self):
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)


# このファイルでconfigの実体をもつ
config = TrainingConfig(
    dataset_root='data',
    batch_size=8,
    num_workers=4,
    learning_rate=1e-4,
    num_epochs = 30,
    image_size=(480, 640),
    in_channels=3,

    rgb_model="nvidia/segformer-b0-finetuned-ade-512-512",
    depth_model="nvidia/segformer-b0-finetuned-ade-512-512",
    dims=(3, 32, 64, 160, 256, 256),
    reductions=(64, 16, 4, 1, 1),
    depths=(1, 1, 1, 1, 1),
    decoder_hiddim=128  # 公式実装のconfigを参照segformer.b0.512x512.ade.160k.py
)