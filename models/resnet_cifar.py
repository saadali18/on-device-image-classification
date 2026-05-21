"""
depth = 6n + 2
ResNet-20 => n = 3
ResNet-32 => n = 5
ResNet-56 => n = 9
ResNet-110 => n = 18
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)
        out = torch.relu(out)

        return out


class ResNetCIFAR(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes: int = 100,
        width: int = 16,
    ):
        super().__init__()

        self.in_planes = width

        self.conv1 = nn.Conv2d(
            3,
            width,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(width)

        self.layer1 = self._make_layer(block, width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, width * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, width * 4, num_blocks[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width * 4 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        layers = []

        strides = [stride] + [1] * (num_blocks - 1)

        for current_stride in strides:
            layers.append(block(self.in_planes, planes, current_stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


def resnet20(num_classes: int = 100, width: int = 16):
    return ResNetCIFAR(
        block=BasicBlock,
        num_blocks=[3, 3, 3],
        num_classes=num_classes,
        width=width,
    )


def resnet32(num_classes: int = 100, width: int = 16):
    return ResNetCIFAR(
        block=BasicBlock,
        num_blocks=[5, 5, 5],
        num_classes=num_classes,
        width=width,
    )


def resnet56(num_classes: int = 100, width: int = 16):
    return ResNetCIFAR(
        block=BasicBlock,
        num_blocks=[9, 9, 9],
        num_classes=num_classes,
        width=width,
    )


def resnet110(num_classes: int = 100, width: int = 16):
    return ResNetCIFAR(
        block=BasicBlock,
        num_blocks=[18, 18, 18],
        num_classes=num_classes,
        width=width,
    )


def build_resnet_cifar(model_name: str, num_classes: int = 100, width: int = 16):
    model_name = model_name.lower()

    if model_name == "resnet20":
        return resnet20(num_classes=num_classes, width=width)

    if model_name == "resnet32":
        return resnet32(num_classes=num_classes, width=width)

    if model_name == "resnet56":
        return resnet56(num_classes=num_classes, width=width)

    if model_name == "resnet110":
        return resnet110(num_classes=num_classes, width=width)

    raise ValueError(f"Unsupported model name: {model_name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)