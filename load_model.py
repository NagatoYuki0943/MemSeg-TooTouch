import torch
from timm import create_model
import yaml
from models import MemSeg
import os
from infer.read_utils import load_dynamic_buffer_module

# ====================================
# Select Model
# ====================================
def load_model(model_dir: str):
    cfg = yaml.load(open(os.path.join(model_dir, 'config.yaml'), 'r'), Loader=yaml.FullLoader)

    # feature_extractor
    feature_extractor = create_model(
        cfg['MODEL']['feature_extractor_name'],
        pretrained    = True,
        features_only = True
    )

    # model
    model = MemSeg(
        feature_extractor = feature_extractor
    )

    model = load_dynamic_buffer_module(model, torch.load(os.path.join(model_dir, 'best_model.pt')))
    return model


if __name__ == "__main__":
    model_dir = "./saved_model/MemSeg-bottle/"
    load_model(model_dir)
