import torch.nn as nn
from .decoder import Decoder
from .msff import MSFF
from .memory_module import MemoryBank


class MemSeg(nn.Module):
    def __init__(self, memory_bank, feature_extractor: nn.Module):
        super().__init__()

        self.memory_bank = memory_bank
        self.feature_extractor = feature_extractor
        self.msff = MSFF()
        self.decoder = Decoder()

    def forward(self, inputs):
        # extract features: [[B, 64, 128, 128], [B, 64, 64, 64], [B, 128, 32, 32], [B, 256, 16, 16], [B, 512, 8, 8]]
        features = self.feature_extractor(inputs)
        f_in = features[0]
        f_out = features[-1]
        f_ii = features[1:-1]

        # extract concatenated information(CI):    [B, 128, 64, 64], [B, 256, 32, 32], [B, 512, 16, 16]
        concat_features = self.memory_bank.forward(features = f_ii)

        # Multi-scale Feature Fusion(MSFF) Module: [B, 64, 64, 64], [B, 128, 32, 32], [B, 256, 16, 16]
        msff_outputs = self.msff(features = concat_features)

        # decoder
        predicted_mask = self.decoder(
            encoder_output  = f_out,                # [B, 512, 8, 8]
            concat_features = [f_in] + msff_outputs # [[B, 64, 128, 128], [B, 64, 64, 64], [B, 128, 32, 32], [B, 256, 16, 16]]
        )

        return predicted_mask # [B, 2, 256, 256]
