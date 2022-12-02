import torch
from torch import Tensor
from torch.utils.data import Dataset

import numpy as np
from typing import List
import warnings


class MemoryBank:
    def __init__(self, normal_dataset: Dataset, nb_memory_sample: int = 30, device='cpu'):

        self.device = device

        # memory bank
        self.memory_information = {}

        # normal dataset
        self.normal_dataset = normal_dataset

        # the number of samples saved in memory bank
        self.nb_memory_sample = nb_memory_sample


    @torch.jit.ignore
    def update(self, feature_extractor):
        feature_extractor.eval()

        # define sample index
        samples_idx = np.arange(len(self.normal_dataset))
        np.random.shuffle(samples_idx)

        # extract features and save features into memory bank
        with torch.no_grad():
            for i in range(self.nb_memory_sample):
                # select image 只选择一部分图片存到memork_bank中
                input_normal, _, _ = self.normal_dataset[samples_idx[i]]
                input_normal = input_normal.to(self.device)

                # extract features
                features = feature_extractor(input_normal.unsqueeze(0))

                # save features into memoery bank 不要 conv1和layer4的输出
                for i, features_l in enumerate(features[1:-1]):
                    if f'level{i}' not in self.memory_information.keys():
                        self.memory_information[f'level{i}'] = features_l
                    else:
                        self.memory_information[f'level{i}'] = torch.cat([self.memory_information[f'level{i}'], features_l], dim=0)


    def _calc_diff(self, features: List[torch.Tensor]) -> torch.Tensor:
        # batch size X the number of samples saved in memory
        diff_bank = torch.zeros(features[0].size(0), self.nb_memory_sample).to(self.device)

        # level
        for l, level in enumerate(self.memory_information.keys()):
            # batch
            for b_idx, features_b in enumerate(features[l]):
                # calculate l2 loss
                diff = mse_loss(
                    input     = torch.repeat_interleave(features_b.unsqueeze(0), repeats=self.nb_memory_sample, dim=0),
                    target    = self.memory_information[level],
                    reduction ='none'
                ).mean(dim=[1, 2, 3])

                # sum loss
                diff_bank[b_idx] += diff

        return diff_bank


    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # calculate difference between features and normal features of memory bank
        diff_bank = self._calc_diff(features=features)

        # concatenate features with minimum difference features of memory bank
        for l, level in enumerate(self.memory_information.keys()):

            selected_features = torch.index_select(self.memory_information[level], dim=0, index=diff_bank.argmin(dim=1))
            diff_features = mse_loss(selected_features, features[l], reduction='none')
            features[l] = torch.cat([features[l], diff_features], dim=1)

        return features


def mse_loss(input: Tensor, target: Tensor, reduction = 'mean'):
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
