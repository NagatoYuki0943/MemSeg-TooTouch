import torch

from timm import create_model
from models import MemSeg, MemoryBank
from data import create_dataset
import os
import logging
import yaml
import argparse

_logger = logging.getLogger('train')


def get_model(cfg, pth: str) -> torch.nn.Module:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # savedir
    cfg['EXP_NAME'] = cfg['EXP_NAME'] + f"-{cfg['DATASET']['target']}"
    savedir = os.path.join(cfg['RESULT']['savedir'], cfg['EXP_NAME'])
    os.makedirs(savedir, exist_ok=True)

    # load memory bank
    memory_bank = torch.load(os.path.join(savedir, f'memory_bank.pt'))
    memory_bank.device = 'cpu'
    for k in memory_bank.memory_information.keys():
        memory_bank.memory_information[k] = memory_bank.memory_information[k].cpu()

    # build feature extractor
    feature_extractor = create_model(
        cfg['MODEL']['feature_extractor_name'],
        pretrained    = True,
        features_only = True
    )

    # build MemSeg
    model = MemSeg(
        memory_bank       = memory_bank,
        feature_extractor = feature_extractor
    )

    # pth model
    if pth:
        state_dict = torch.load(pth)
        model.load_state_dict(state_dict)
    model.to(device)
    return model


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='test memseg')
    parser.add_argument('--yaml_config', type=str, default=None, help='exp config file')
    parser.add_argument('--pth', type=str, default=None, help='model pth file')

    args = parser.parse_args()

    # config
    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)

    model = get_model(cfg, args.pth)
    for name, children in model.named_children():
        print(name)
