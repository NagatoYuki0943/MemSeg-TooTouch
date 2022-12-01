import torch
from timm import create_model
from models import MemSeg
import yaml
import os


# ====================================
# Select Model
# ====================================
def load_model(cfg, model_dir):

    # memory_bank
    memory_bank = torch.load(os.path.join(model_dir, 'memory_bank.pt'))
    memory_bank.device = 'cpu'
    for k in memory_bank.memory_information.keys():
        memory_bank.memory_information[k] = memory_bank.memory_information[k].cpu()

    # feature_extractor
    feature_extractor = create_model(
        cfg['MODEL']['feature_extractor_name'],
        pretrained    = True,
        features_only = True
    )

    # model
    model = MemSeg(
        memory_bank       = memory_bank,
        feature_extractor = feature_extractor
    )

    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pt')))
    return model


if __name__=='__main__':
    model_dir  = "./saved_model/mvtec/MemSeg-bottle"
    # load config
    cfg = yaml.load(open(os.path.join(model_dir, "config.yaml"),'r'), Loader=yaml.FullLoader)
    model = load_model(cfg, model_dir)
