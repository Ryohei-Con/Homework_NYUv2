import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientAttention(nn.Module):
    def __init__(self, dim, reduction, num_heads=4, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % self.num_heads == 0, "dim must be divisible by num_heads"

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, 2*dim, bias=False)
        self.scale = (dim // num_heads) ** -0.5
        self.proj = nn.Linear(dim, dim, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.reduction = reduction
        if reduction > 1:
            self.sr = nn.Conv2d(dim, dim, reduction, reduction)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, h, w):
        """
        x: (Batch, N, C)
        """
        b, n, c = x.size()

        print(f"x-size at attention\n{x.size()}")
        print(f"{h=}\n{w=}")

        # -> (B, H, N, head_dim)
        # full resolution
        q: torch.Tensor = self.to_q(x).reshape(b, n, self.num_heads, c//self.num_heads).permute(0, 2, 1, 3)

        # -> (B, H, reduced_N, head_dim)
        # spatially reduced
        if self.reduction > 1:
            x = x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            x = self.sr(x)
            x = x.permute(0, 2, 3, 1).reshape(b, -1, c)
            x = self.norm(x)
            kv = self.to_kv(x).reshape(b, -1, 2, self.num_heads, c//self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv: torch.Tensor = self.to_kv(x).reshape(b, -1, 2, self.num_heads, c//self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # -> (B, H, N, reduced_N)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, -1)
        attn = self.attn_drop(attn)
        # -> (B, H, N, head_dim)
        output = torch.matmul(attn, v)
        output = output.permute(0, 2, 1, 3).reshape(b, n, -1)
        output = self.proj(output)
        output = self.proj_drop(output)
        return output  # (B, N, C)


class MixFFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.first_mlp = nn.Linear(dim, 4 * dim)
        self.conv = DWConv(4 * dim)
        self.gelu = nn.GELU()
        self.last_mlp = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor, h, w):
        """Mix FFN

        Args:
            x (torch.Tensor): (Batch, Length, emb_dim)
        """
        b, l, dim = x.size()
        x = self.first_mlp(x)
        x = self.conv(x, h, w)
        x = self.gelu(x)
        x = self.last_mlp(x)
        x = x.reshape(b, l, dim)
        return x  # (Batch, Length, emb_dim)


class DWConv(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.dwConv = nn.Conv2d(hid_dim, hid_dim, kernel_size=3, padding=1, groups=hid_dim)

    def forward(self, x, h, w):
        """
        x: (Batch, Length, emb_dim)
        """
        b, l, emb_dim = x.size()
        x = x.reshape(b, h, w, emb_dim)
        x = x.permute(0, 3, 1, 2)
        x = self.dwConv(x)
        x = x.permute(0, 2, 3, 1)
        return x  # (Batch, Length, emb_dim)


class OverlapPatchMerging(nn.Module):
    def __init__(self, in_channel, out_channel, padding, kernel, stride):
        super().__init__()
        self.proj = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.norm = nn.LayerNorm([out_channel])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """
        Args:
            x (torch.Tensor): (Batch, emb_dim, H, W)
        """
        print(f"In OverlapPatchMerging(before projection)\n{x.size()=}")
        x = self.proj(x)  # (Batch, emb_dim, H, W)
        b, d, h, w = x.size()
        print(f"In OverlapPatchMerging(after projection)\n{x.size()}\n{h=}\n{w=}")
        x = x.flatten(2).transpose(1, 2)  # (B, N, emb_dim)
        x = self.norm(x)
        return (x, h, w)  # (B, N, emb_dim)


class Block(nn.Module):
    def __init__(self, dim, reduction):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = EfficientAttention(dim, reduction)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, h, w):
        """
        args:
        x torch.Tensor: (Batch, Length, Emb_dim)

        returns torch.Tensor: (Batch, Length, Emb_dim)

        """
        x = self.dropout(self.attention(self.norm1(x), h, w)) + x
        x = self.dropout(self.mlp(self.norm2(x), h, w)) + x
        return x
