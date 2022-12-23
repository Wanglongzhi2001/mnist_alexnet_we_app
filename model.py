import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, pad_mode="valid", has_bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                     has_bias=has_bias, pad_mode=pad_mode)


def fc_with_initialize(input_channels, out_channels, has_bias=True):
    return nn.Dense(input_channels, out_channels, has_bias=has_bias)


class AlexNet(nn.Cell):
    """
    Alexnet
    """

    def __init__(self, num_classes=10, channel=1, phase='train', dataset_name='imagenet'):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(channel, 6, 5, pad_mode='valid')
        self.batchnorm1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.conv3 = conv(16, 16, 3, stride=1, pad_mode="pad", padding=1, has_bias=True)
        self.batchnorm3 = nn.BatchNorm2d(16)
        self.conv4 = conv(16, 16, 3, stride=1, pad_mode="pad", padding=1, has_bias=True)
        self.batchnorm4 = nn.BatchNorm2d(16)
        self.conv5 = conv(16, 32, 3, stride=1, pad_mode="pad", padding=1, has_bias=True)
        self.batchnorm5 = nn.BatchNorm2d(32)
        self.relu = P.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.flatten = nn.Flatten()
        self.fc1 = fc_with_initialize(128, 256)
        self.fc2 = fc_with_initialize(256, 512)
        self.fc3 = fc_with_initialize(512, num_classes)
        self.sm = nn.Softmax()
        self.dropout = nn.Dropout()

    def construct(self, x):
        """define network"""
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.relu(x)
        x = self.max_pool2d(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sm(x)
        return x