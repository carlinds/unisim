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

from typing import Tuple

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn


class OccupancyGrid(nn.Module):
    """Data to represent the occupancy grid."""

    def __init__(
        self,
        voxel_size: float,
        aabb: Float[Tensor, "2 3"],
        dilation_kernel_size: int = 0,
    ):
        """Initializes the occupancy grid.

        Args:
            voxel_size: the size of the voxels
            aabb: the axis-aligned bounding box (aabb[0] is the minimum (x,y,z) point, aabb[1] is the maximum (x,y,z) point)
            dilation_kernel_size: the size of the dilation kernel
        """
        super().__init__()

        self.register_buffer("voxel_size", torch.tensor(voxel_size))
        aabb[1] = aabb[0] + ((aabb[1] - aabb[0]) / voxel_size).ceil() * voxel_size
        self.register_buffer("aabb", aabb)
        grid = self._create_grid()
        self.register_buffer("grid", grid)
        self.dilation_kernel_size = dilation_kernel_size

    def _create_grid(self):
        """Creates an empty occupancy grid."""

        # create the occupancy grid
        occupancy_grid = torch.zeros(
            ((self.aabb[1] - self.aabb[0]) / self.voxel_size).ceil().int().tolist(),
            dtype=torch.bool,
        )
        return occupancy_grid

    def populate_occupancy(self, points: Float[Tensor, "*batch 3"]):
        """Populates the occupancy grid with the given points."""
        points = points.to(self.aabb.device)
        voxel_indices, is_in_bounds = self.get_voxel_indices(points)
        voxel_indices = voxel_indices[is_in_bounds]
        self.grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = True

        if self.dilation_kernel_size > 0:
            self.dilate_grid(self.dilation_kernel_size)

    def get_voxel_indices(
        self, points: Float[Tensor, "*batch 3"]
    ) -> Tuple[Int[Tensor, "*batch 3"], Bool[Tensor, " *batch"]]:
        """Returns the voxel indices for the points."""
        # get the voxel indices
        voxel_indices_float = (points - self.aabb[0]) / self.voxel_size
        max_idxs = torch.tensor(self.grid.shape, device=self.grid.device) - 1
        is_in_bounds = ((voxel_indices_float >= 0) & (voxel_indices_float <= max_idxs)).all(dim=-1)
        voxel_indices_float[~is_in_bounds] = 0
        return voxel_indices_float.long(), is_in_bounds

    def is_occupied(
        self, points: Float[Tensor, "*batch 3"], invalid_is_occupied: bool = False
    ) -> Bool[Tensor, " *batch"]:
        """Returns whether the points are in occupied cell."""
        voxel_indices, is_valid = self.get_voxel_indices(points)
        if invalid_is_occupied:
            return ~is_valid | self.grid[voxel_indices[..., 0], voxel_indices[..., 1], voxel_indices[..., 2]]
        else:
            return is_valid & self.grid[voxel_indices[..., 0], voxel_indices[..., 1], voxel_indices[..., 2]]

    def dilate_grid(self, kernel_size: int):
        """Dilates the grid with the given kernel size."""
        kernel = torch.ones((kernel_size, kernel_size, kernel_size), dtype=torch.float32, device=self.grid.device)
        pad_right = kernel_size // 2
        pad_left = pad_right - 1 if (kernel_size % 2) == 0 else pad_right
        padding = tuple([pad_left, pad_right] * 3)
        grid = torch.nn.functional.pad(self.grid[None, None], padding)
        if (device := grid.device).type == "mps":  # MPS does not support conv3d, so we run this on the CPU
            grid, kernel = grid.to("cpu"), kernel.to("cpu")
        grid = torch.nn.functional.conv3d(grid.float(), kernel[None, None])[0, 0].bool()
        if device.type == "mps":
            grid = grid.to(device)
        self.grid = grid
