from torch import Tensor
import torch.nn as nn
from typing import List
from experiment_config import DatasetType


class ExperimentBaseModel(nn.Module):
    def __init__(self, dataset_type: DatasetType):
        super().__init__()
        self.dataset_type = dataset_type

    def forward(self, x) -> Tensor:
        raise NotImplementedError


class NiNBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, batch_norm: bool, dropout_prob: float) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(planes) if batch_norm else lambda x: x
        self.dp1 = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else lambda x: x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dp1(x)

        return x


class NiN(ExperimentBaseModel):
    def __init__(self, depth: int, width: int, base_width: int, batch_norm_layers: List[int], dropout_prob: float,
                 dataset_type: DatasetType) -> None:
        super().__init__(dataset_type)
        batch_norm_layers = [] if batch_norm_layers is None else batch_norm_layers

        self.base_width = base_width

        blocks = []
        blocks.append(NiNBlock(self.dataset_type.D[0], self.base_width * width, 0 in batch_norm_layers, dropout_prob))
        for i in range(depth - 1):
            blocks.append(NiNBlock(self.base_width * width, self.base_width * width, i in batch_norm_layers,
                                   dropout_prob))
        self.blocks = nn.Sequential(*blocks)

        self.conv = nn.Conv2d(self.base_width * width, self.dataset_type.K, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(self.dataset_type.K)
        self.dp = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else lambda x: x
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.blocks(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dp(x)

        x = self.avgpool(x)

        return x.squeeze()
