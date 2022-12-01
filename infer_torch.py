import torch
from torch.nn import functional as F
import time
import os
from statistics import mean
from timm import create_model
import numpy as np
import cv2
from models import MemSeg
from infer.infer import Inference
from infer.read_utils import *


class TorchInference(Inference):
    def __init__(self, yaml_path: str, model_dir: str, use_cuda: bool=False) -> None:
        """
        Args:
            yaml_path (str):  配置文件路径
            model_dir (str):  模型文件夹路径
            use_cuda (bool, optional): 是否使用cuda. Defaults to False.
        """
        super().__init__()
        self.use_cuda = use_cuda
        # 超参数
        self.config = get_yaml_config(yaml_path)
        self.infer_size = self.config['DATASET']['resize']
        # 载入模型
        self.model = self.get_model(model_dir)
        self.model.eval()
        # 预热模型
        self.warm_up()


    def get_model(self, model_dir: str):
        """获取script模型

        Args:
            torchscript_path (str): 模型路径

        Returns:
            torchscript: script模型
        """
        # memory_bank
        # memory_bank = torch.load(os.path.join(model_dir, 'memory_bank.pt'))
        # if self.use_cuda:
        #     memory_bank.device = 'cuda'
        #     for k in memory_bank.memory_information.keys():
        #         memory_bank.memory_information[k] = memory_bank.memory_information[k].cuda()
        # else:
        #     memory_bank.device = 'cpu'
        #     for k in memory_bank.memory_information.keys():
        #         memory_bank.memory_information[k] = memory_bank.memory_information[k].cpu()

        # feature_extractor
        feature_extractor = create_model(
            self.config['MODEL']['feature_extractor_name'],
            pretrained    = True,
            features_only = True
        )

        # model
        model = MemSeg(
            # memory_bank       = memory_bank,
            feature_extractor = feature_extractor
        )

        model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pt')))
        if self.use_cuda:
            model = model.cuda()
        return model


    def warm_up(self):
        """预热模型
        """
        infer_height, infer_width = self.infer_size
        x = torch.zeros(1, 3, infer_height, infer_width)
        if self.use_cuda:
            x = x.cuda()
        with torch.inference_mode():
            self.model(x)


    def infer(self, image: np.ndarray) -> np.ndarray:
        """推理单张图片

        Args:
            image (np.ndarray): 图片

        Returns:
            np.ndarray: hotmap
        """
        # 1.保存图片高宽
        image_height, image_width = image.shape[0], image.shape[1]

        # 2.图片预处理
        # 推理时使用的图片大小
        infer_height, infer_width = self.infer_size
        transform = get_transform(infer_height, infer_width, tensor=True)
        x = transform(image=image)
        x = x['image'].unsqueeze(0)

        # 3.预测得到热力图
        if self.use_cuda:
            x = x.cuda()
        with torch.inference_mode():
            y = self.model(x)     # 单个值返回Tensor,多个值返回tuple

        # 4.预测后处理
        y = F.softmax(y, dim=1) # [1, 2, H, W]
        y = y[0][1]             # [H, W] 取出1,代表有问题的层
        y = y.squeeze()         # [H, W]
        y = y.detach().cpu().numpy()

        # 5.还原到原图尺寸
        y = cv2.resize(y, (image_width, image_height))

        return y


def single(yaml_path: str, model_dir: str, image_path: str, save_path: str, use_cuda: bool = False) -> None:
    """预测单张图片

    Args:
        yaml_path (str):    配置文件路径
        model_dir (str):    模型文件夹
        image_path (str):   图片路径
        save_path (str):    保存图片路径
        use_cuda (bool, optional): 是否使用cuda. Defaults to False.
    """
    # 1.获取推理器
    inference = TorchInference(yaml_path, model_dir, use_cuda)

    # 2.打开图片
    image = load_image(image_path)

    # 3.推理
    start = time.time()
    anomaly_map = inference.infer(image)    # [900, 900]

    # 4.生成mask,mask边缘,热力图叠加原图
    mask, mask_outline, superimposed_map = gen_images(image, anomaly_map)
    end = time.time()

    print("infer time:", end - start)

    # 5.保存图片
    save_image(save_path, image, mask, mask_outline, superimposed_map)


def multi(yaml_path: str, model_dir: str, image_dir: str, save_dir: str, use_cuda: bool = False) -> None:
    """预测多张图片

    Args:
        yaml_path (str):    配置文件路径
        model_dir (str):    模型文件夹
        image_dir (str):    图片文件夹
        save_dir (str, optional):  保存图片路径,没有就不保存. Defaults to None.
        use_cuda (bool, optional): 是否使用cuda. Defaults to False.
    """
    # 0.检查保存路径
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"mkdir {save_dir}")
    else:
        print("保存路径为None,不会保存图片")

    # 1.获取推理器
    inference = TorchInference(yaml_path, model_dir, use_cuda)

    # 2.获取文件夹中图片
    imgs = os.listdir(image_dir)
    imgs = [img for img in imgs if img.endswith(("jpg", "jpeg", "png", "bmp"))]

    infer_times: list[float] = []
    # 批量推理
    for img in imgs:
        # 3.拼接图片路径
        image_path = os.path.join(image_dir, img);

        # 4.打开图片
        image = load_image(image_path)

        # 5.推理
        start = time.time()
        anomaly_map = inference.infer(image)    # [900, 900]

        # 6.生成mask,mask边缘,热力图叠加原图
        mask, mask_outline, superimposed_map = gen_images(image, anomaly_map)
        end = time.time()

        infer_times.append(end - start)
        print("infer time:", end - start)

        if save_dir is not None:
            # 7.保存图片
            save_path = os.path.join(save_dir, img)
            save_image(save_path, image, mask, mask_outline, superimposed_map)

    print("avg infer time: ", mean(infer_times))


if __name__ == "__main__":
    image_path = "./datasets/MVTec/bottle/test/broken_large/000.png"
    image_dir  = "./datasets/MVTec/bottle/test/broken_large"
    yaml_path  = "./saved_model/MemSeg-bottle/config.yaml"
    model_dir  = "./saved_model/MemSeg-bottle"
    save_path  = "./saved_model/MemSeg-bottle/torch_output.jpg"
    save_dir   = "./saved_model/MemSeg-bottle/result"
    # single(yaml_path, model_dir, image_path, save_path, use_cuda = True)
    # multi(yaml_path, model_dir, image_dir, save_dir, use_cuda = True)

    image_path = "./datasets/custom/test/bad/001.jpg"
    image_dir  = "./datasets/custom/test/bad"
    yaml_path  = "./saved_model/1/MemSeg-custom-256/config.yaml"
    model_dir  = "./saved_model/1/MemSeg-custom-256"
    save_path  = "./saved_model/1/MemSeg-custom-256/torch_output.jpg"
    save_dir   = "./saved_model/1/MemSeg-custom-256/result"
    single(yaml_path, model_dir, image_path, save_path, use_cuda = True)
    # multi(yaml_path, model_dir, image_dir, save_dir, use_cuda = True)
