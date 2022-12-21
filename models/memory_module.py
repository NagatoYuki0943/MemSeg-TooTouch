import torch
from torch import Tensor, nn
import torch.nn.functional as F

import numpy as np
from typing import List
import warnings


class MemoryBank:
    def __init__(self, normal_dataset, nb_memory_sample: int = 30, device='cpu'):
        self.device = device

        # memory_information
        # {
        #   'level0': [30,  64, 64, 64],
        #   'level1': [30, 128, 32, 32],
        #   'level2': [30, 256, 16, 16]
        # }
        self.memory_information = {}

        # normal dataset
        self.normal_dataset = normal_dataset

        # the number of samples saved in memory_information
        self.nb_memory_sample = nb_memory_sample


    def update(self, feature_extractor: nn.Module):
        """使用feature_extractor提取正常图片的特征,放入memory_information中

        Args:
            feature_extractor (nn.Module): 特征提取器
        """
        feature_extractor.eval()

        # define sample index 定义随机的index
        samples_idx = np.arange(len(self.normal_dataset))
        np.random.shuffle(samples_idx)

        # extract features and save features into memory_information
        # 提取30张正常图片的信息放入memorybank中
        with torch.no_grad():
            for i in range(self.nb_memory_sample):
                # select image
                input_normal, _, _ = self.normal_dataset[samples_idx[i]]
                input_normal = input_normal.to(self.device)

                # extract features
                features = feature_extractor(input_normal.unsqueeze(0))

                # save features into memory_information
                for i, features_l in enumerate(features[1:-1]):
                    if f'level{i}' not in self.memory_information.keys():
                        self.memory_information[f'level{i}'] = features_l
                    else:
                        self.memory_information[f'level{i}'] = torch.cat([self.memory_information[f'level{i}'], features_l], dim=0)


    def _calc_diff(self, features: List[torch.Tensor]) -> torch.Tensor:
        """对比新特征和memory_information中的差异

        Args:
            features (List[torch.Tensor]): feature_extractor提取的第1,2,3层信息 [[B, 64, 64, 64], [B, 128, 32, 32], [B, 256, 16, 16]]

        Returns:
            torch.Tensor: 对比差异得到的loss [B, 30]
        """
        # batch size X the number of samples saved in memory
        # [B, 30]
        diff_bank = torch.zeros(features[0].size(0), self.nb_memory_sample).to(self.device)

        # level
        for l, level in enumerate(self.memory_information.keys()):
            # batch                            features[l]取出某一层输出,再通过for循环对batch进行循环
            for b_idx, features_b in enumerate(features[l]):
                # calculate l2 loss mse_loss(([C, H, W] -> [30, C, H, W]), [30, C, H, W]) = [30, C, H, W] -> [30]
                diff = mse_loss(
                    input     = torch.repeat_interleave(features_b.unsqueeze(0), repeats=self.nb_memory_sample, dim=0),
                    target    = self.memory_information[level],
                    reduction ='none'
                ).mean(dim=[1,2,3])

                # sum loss
                diff_bank[b_idx] += diff

        return diff_bank


    def select(self, features: List[torch.Tensor]) -> torch.Tensor:
        """对feature_extractor提取的新特征与memory_information对比计算得到结果与输入拼接返回

        Args:
            features (List[torch.Tensor]): feature_extractor提取的第1,2,3层信息 [[B, 64, 64, 64], [B, 128, 32, 32], [B, 256, 16, 16]]

        Returns:
            torch.Tensor: 拼接features和loss的tensor
        """
        # calculate difference between features and normal features of memory_information
        diff_bank = self._calc_diff(features=features) # [[B, 64, 64, 64], [B, 128, 32, 32], [B, 256, 16, 16]] -> [B, 30]

        # concatenate features with minimum difference features of memory_information
        for l, level in enumerate(self.memory_information.keys()):
            # 在memory_information中选择和当前feature差距最小的,形状和feature相同,like [B, 64, 64, 64], [B, 128, 32, 32], [B, 256, 16, 16]
            selected_features = torch.index_select(self.memory_information[level], dim=0, index=diff_bank.argmin(dim=1))
            # 求选择的features和当前features的loss,输出形状和feature相同,like [B, 64, 64, 64], [B, 128, 32, 32], [B, 256, 16, 16]
            diff_features = mse_loss(selected_features, features[l], reduction='none')
            # [B, C, H, W] cat [B, C, H, W] = [B, 2*C, H, W]
            features[l] = torch.cat([features[l], diff_features], dim=1)

        return features # [[B, 128, 64, 64], [B, 256, 32, 32], [B, 512, 16, 16]]


def mse_loss(input: Tensor, target: Tensor, reduction = 'mean') -> Tensor:
    """mse loss
    input and target shape should be same.

    Args:
        input (Tensor):  predict value
        target (Tensor): target value
        reduction (str, optional): mean' | 'sum' | 'none'. Defaults to 'mean'.

    Returns:
        Tensor: mse result
    """
    if target.size() != input.size():
        warnings.warn(
            "Using a target size ({}) that is different to the input size ({}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.".format(target.size(), input.size()),
            stacklevel=2,
        )

    result: Tensor = (input - target) ** 2
    if reduction == "mean":
        return result.mean()
    elif reduction == "sum":
        return result.sum()
    elif reduction == "none":
        return result
