import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
from typing import List


class MemoryBank(nn.Module):
    def __init__(self):
        super().__init__()

        self.device: str

        # memory bank
        # self.memory_information = {}
        self.register_buffer("layer1", torch.Tensor(0))
        self.register_buffer("layer2", torch.Tensor(0))
        self.register_buffer("layer3", torch.Tensor(0))
        self.layer1: torch.Tensor
        self.layer2: torch.Tensor
        self.layer3: torch.Tensor

    @torch.jit.ignore
    def update(self, feature_extractor, normal_dataset: Dataset, nb_memory_sample: int = 30, device='cpu'):
        """update layer1,2,3

        Args:
            feature_extractor (nn.Module): resnet
            normal_dataset (Dataset): normaldataset
            nb_memory_sample (int, optional): the number of samples saved in memory bank. Defaults to 30.
        """
        self.nb_memory_sample = nb_memory_sample
        self.device = device

        memory_information = {}

        feature_extractor.eval()

        # define sample index
        samples_idx = np.arange(len(normal_dataset))
        np.random.shuffle(samples_idx)

        # extract features and save features into memory bank
        with torch.no_grad():
            for i in range(nb_memory_sample):
                # select image 只选择一部分图片存到memork_bank中
                input_normal, _, _ = normal_dataset[samples_idx[i]]
                input_normal = input_normal.to(self.device)

                # extract features
                features = feature_extractor(input_normal.unsqueeze(0))

                # save features into memoery bank 不要 conv1和layer4的输出
                for i, features_l in enumerate(features[1:-1]):
                    if i not in memory_information.keys():
                        memory_information[i] = features_l
                    else:
                        memory_information[i] = torch.cat([memory_information[i], features_l], dim=0)

        # save to register buffer
        self.layer1 = memory_information[0] # [nb_memory_sample,  64, 64, 64]
        self.layer2 = memory_information[1] # [nb_memory_sample, 128, 32, 32]
        self.layer3 = memory_information[2] # [nb_memory_sample, 256, 16, 16]


    def _calc_diff(self, features: List[torch.Tensor]) -> torch.Tensor:
        # batch size X the number of samples saved in memory
        diff_bank = torch.zeros(features[0].size(0), self.nb_memory_sample).to(self.device)

        # level
        for l in range(3):
            # 获取对应的memory
            memory_information = {
                0: self.layer1,
                1: self.layer2,
                2: self.layer3,
            }[l]
            # batch
            for b_idx, features_b in enumerate(features[l]):
                # calculate l2 loss
                diff = F.mse_loss(
                    input     = torch.repeat_interleave(features_b.unsqueeze(0), repeats=self.nb_memory_sample, dim=0),
                    target    = memory_information,
                    reduction ='none'
                ).mean(dim=[1, 2, 3])

                # sum loss
                diff_bank[b_idx] += diff

        return diff_bank


    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # calculate difference between features and normal features of memory bank
        diff_bank = self._calc_diff(features=features)

        # concatenate features with minimum difference features of memory bank
        for l in range(3):
            # 获取对应的memory
            memory_information = {
                0: self.layer1,
                1: self.layer2,
                2: self.layer3,
            }[l]

            selected_features = torch.index_select(memory_information, dim=0, index=diff_bank.argmin(dim=1))
            diff_features = F.mse_loss(selected_features, features[l], reduction='none')
            features[l] = torch.cat([features[l], diff_features], dim=1)

        return features
