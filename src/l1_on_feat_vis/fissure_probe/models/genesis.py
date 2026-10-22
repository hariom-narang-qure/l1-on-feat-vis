"""Models Genesis 3D U-Net (Zhou et al., MICCAI 2019), chest CT weights.

Re-implemented from the published architecture so the checkpoint loads with
strict key matching, without running the authors' code:
  - LUConv = Conv3d(k=3, p=1) -> BatchNorm3d -> ReLU.
  - Encoder block `down_trN` = two LUConvs; blocks 0-2 are followed by a
    2x max-pool, and their pre-pool output is the skip connection.
  - Decoder block `up_trN` = ConvTranspose3d(k=2, s=2), concat the skip, two LUConvs.
  - Input: (N, 1, X, Y, Z), HU clipped to [-1000, 1000] and scaled to [0, 1],
    spatial sizes divisible by 8 (three pools).
Weights: https://huggingface.co/MrGiovanni/ModelsGenesis (Genesis_Chest_CT.pt).
"""

from pathlib import Path

import torch
from torch import nn

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[4] / "weights" / "Genesis_Chest_CT.pt"

# Block outputs we probe, from input resolution to bottleneck and back.
PROBED_BLOCKS = ["down_tr64", "down_tr128", "down_tr256", "down_tr512", "up_tr256", "up_tr128", "up_tr64"]
BLOCK_STRIDE = {"down_tr64": 1, "down_tr128": 2, "down_tr256": 4, "down_tr512": 8,
                "up_tr256": 4, "up_tr128": 2, "up_tr64": 1}


class LUConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn1(self.conv1(x)))


def double_conv(in_channels, depth):
    first = 32 * 2 ** depth
    return nn.Sequential(LUConv(in_channels, first), LUConv(first, first * 2))


class DownTransition(nn.Module):
    def __init__(self, in_channels, depth):
        super().__init__()
        self.ops = double_conv(in_channels, depth)
        self.pool = nn.MaxPool3d(2) if depth < 3 else nn.Identity()

    def forward(self, x):
        skip = self.ops(x)
        return self.pool(skip), skip


class UpTransition(nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super().__init__()
        self.up_conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.ops = nn.Sequential(
            LUConv(in_channels + out_channels // 2, 32 * 2 ** (depth + 1)),
            LUConv(32 * 2 ** (depth + 1), 32 * 2 ** (depth + 1)),
        )

    def forward(self, x, skip):
        return self.ops(torch.cat([self.up_conv(x), skip], dim=1))


class OutputTransition(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.final_conv = nn.Conv3d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        return torch.sigmoid(self.final_conv(x))


class GenesisUNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_tr64 = DownTransition(1, 0)
        self.down_tr128 = DownTransition(64, 1)
        self.down_tr256 = DownTransition(128, 2)
        self.down_tr512 = DownTransition(256, 3)
        self.up_tr256 = UpTransition(512, 512, 2)
        self.up_tr128 = UpTransition(256, 256, 1)
        self.up_tr64 = UpTransition(128, 128, 0)
        self.out_tr = OutputTransition(64)

    def forward_features(self, x):
        """Every probed block's output (post-ReLU), keyed by block name.

        For encoder blocks this is the pre-pool output: the same tensor the
        skip connection carries, at that block's own resolution.
        """
        features = {}
        x, features["down_tr64"] = self.down_tr64(x)
        x, features["down_tr128"] = self.down_tr128(x)
        x, features["down_tr256"] = self.down_tr256(x)
        _, features["down_tr512"] = self.down_tr512(x)
        x = features["up_tr256"] = self.up_tr256(features["down_tr512"], features["down_tr256"])
        x = features["up_tr128"] = self.up_tr128(x, features["down_tr128"])
        features["up_tr64"] = self.up_tr64(x, features["down_tr64"])
        return features

    def forward(self, x):
        return self.out_tr(self.forward_features(x)["up_tr64"])


def strip_data_parallel_prefix(state_dict):
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_genesis(weights=DEFAULT_WEIGHTS, pretrained=True, seed=0):
    """pretrained=False gives the same architecture with random init: the
    baseline for "does pretraining matter, or does any 3D conv net see this?"."""
    torch.manual_seed(seed)
    model = GenesisUNet3D()
    if pretrained:
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(strip_data_parallel_prefix(checkpoint["state_dict"]), strict=True)
    return model.eval()
