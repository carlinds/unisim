# Copyright 2024 the authors of NeuRAD and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from nerfstudio.model_components.cnns import BasicBlock
from torch import nn


class PatchGANDiscriminator(nn.Module):
    """Convoliutional discriminator network, used for adversarial loss."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=2),  # 96 -> 49
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=2),  # 49 -> 25
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=2),  # 25 -> 13
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=2),  # 13 -> 7
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # reduce to 1x1 patch
            nn.Conv2d(512, 1, kernel_size=5, stride=1, padding=0),  # 7 -> 3
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),  # 1x1 -> 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_loss(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """Compute the loss for the discriminator.

        Args:
            x: Input tensor. Assumes half of the batch is real and half is fake.

        Returns:
            Loss tensor.
        """
        out_real = self.forward(real.detach().permute(0, 3, 1, 2))
        out_fake = self.forward(fake.detach().permute(0, 3, 1, 2))
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([out_real, out_fake]),
            torch.cat([torch.ones_like(out_real), torch.zeros_like(out_fake)]),
        )
        return loss


class ConvDiscriminator(nn.Module):
    """Convolutional discriminator network, used for adversarial loss."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            BasicBlock(3, 64, kernel_size=3, padding=1),
            nn.AvgPool2d(2),  # 64 -> 32
            BasicBlock(64, 128, kernel_size=3, padding=1),
            nn.AvgPool2d(2),  # 32 -> 16
            BasicBlock(128, 192, kernel_size=3, padding=1),
            nn.AvgPool2d(2),  # 16 -> 8
            BasicBlock(192, 256, kernel_size=3, padding=1),
            nn.AvgPool2d(2),  # 8 -> 4
            BasicBlock(256, 352, kernel_size=3, padding=1),
            nn.AvgPool2d(2),  # 4 -> 2
            BasicBlock(352, 448, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1),  # 2 -> 1
            nn.Flatten(),
            nn.Linear(448, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_loss(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """Compute the loss for the discriminator.

        Args:
            x: Input tensor. Assumes half of the batch is real and half is fake.

        Returns:
            Loss tensor.
        """
        out_real = self.forward(real.detach().permute(0, 3, 1, 2))
        out_fake = self.forward(fake.detach().permute(0, 3, 1, 2))
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([out_real, out_fake]),
            torch.cat([torch.ones_like(out_real), torch.zeros_like(out_fake)]),
        )
        return loss
