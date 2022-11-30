import torch
import cv2
import numpy as np
from typing import Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm import create_model
from models import MemSeg
from torch.nn import functional as F
import yaml
import os
from skimage import morphology
from skimage.morphology import dilation
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt


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


#-----------------------------#
#   打开图片
#-----------------------------#
def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    # print(isinstance(image, np.ndarray))              # True
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # BGR2RGB
    return image


#-----------------------------#
#   图片预处理
#   支持pytorch和numpy
#-----------------------------#
def get_transform(height: int, width: int, tensor = True) -> Callable:
    """图片预处理,支持pytorch和numpy

    Args:
        height (int): 缩放的高
        width (int):  缩放的宽
        tensor (bool, optional): pytorch or numpy. Defaults to True.
    """
    mean = np.array((0.485, 0.456, 0.406))
    std  = np.array((0.229, 0.224, 0.225))
    if tensor:
        return A.Compose(
            [
                A.Resize(height=height, width=width, always_apply=True),
                A.Normalize(mean=mean, std=std), # 归一化+标准化
                ToTensorV2(),
            ]
        )
        # return transforms.Compose(
        #     [
        #         transforms.Resize((height, width)), # torchvision的Resize只支持 PIL Image,不支持 numpy.ndarray, opencv读取的图片就是 .ndarray 格式
        #         transforms.ToTensor(),              # 转化为tensor,并归一化
        #         transforms.Normalize(mean, std)     # 减去均值(前面),除以标准差(后面)
        #     ]
        # )
    else:
        def transform(image: np.ndarray) -> np.ndarray:
            image = cv2.resize(image, [width, height])
            image = image.astype(np.float32)
            image /= 255.0
            image -= mean
            image /= std
            image = image.transpose(2, 0, 1)    # [h, w, c] -> [c, h, w]
            return {"image": image}
        return transform




#-----------------------------#
#   单通道热力图转换为rgb
#-----------------------------#
def anomaly_map_to_color_map(anomaly_map: np.ndarray, normalize: bool = True) -> np.ndarray:
    """ 单通道热力图转换为rgb
        Compute anomaly color heatmap.

    Args:
        anomaly_map (np.ndarray): Final anomaly map computed by the distance metric.        [900, 900]
        normalize (bool, optional): Bool to normalize the anomaly map prior to applying
            the color map. Defaults to True.

    Returns:
        np.ndarray: [description]                                                           [900, 900, 3]
    """
    if normalize:
        anomaly_map = (anomaly_map - anomaly_map.min()) / np.ptp(anomaly_map)   # np.ptp()函数实现的功能等同于np.max(array) - np.min(array)
    anomaly_map = anomaly_map * 255                                             # 0~1 -> 0~255
    anomaly_map = anomaly_map.astype(np.uint8)                                  # 变为整数

    anomaly_map = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)              # [900, 900] -> [900, 900, 3]
    anomaly_map = cv2.cvtColor(anomaly_map, cv2.COLOR_RGB2BGR)                  # RGB2BGR
    return anomaly_map


#-----------------------------#
#   将热力图和原图叠加
#-----------------------------#
def superimpose_anomaly_map(
    anomaly_map: np.ndarray, image: np.ndarray, alpha: float = 0.4, gamma: int = 0, normalize: bool = False
) -> np.ndarray:
    """将热力图和原图叠加
        Superimpose anomaly map on top of in the input image.

    Args:
        anomaly_map (np.ndarray): Anomaly map       热力图  [900, 900]
        image (np.ndarray): Input image             原图    [900, 900]
        alpha (float, optional): Weight to overlay anomaly map
            on the input image. Defaults to 0.4.
        gamma (int, optional): Value to add to the blended image
            to smooth the processing. Defaults to 0. Overall,
            the formula to compute the blended image is
            I' = (alpha*I1 + (1-alpha)*I2) + gamma
        normalize: whether or not the anomaly maps should
            be normalized to image min-max

    Returns:
        np.ndarray: Image with anomaly map superimposed on top of it.
    """

    # 单通道热力图转换为rgb [900, 900] -> [900, 900, 3]
    anomaly_map = anomaly_map_to_color_map(anomaly_map.squeeze(), normalize=normalize)
    # 叠加图片
    superimposed_map = cv2.addWeighted(anomaly_map, alpha, image, (1 - alpha), gamma)
    return superimposed_map


# from anomalib.post_processing.post_process import compute_mask
def compute_mask(anomaly_map: np.ndarray, threshold: float = 0.5, kernel_size: int = 1) -> np.ndarray:
    """Compute anomaly mask via thresholding the predicted anomaly map.

    Args:
        anomaly_map (np.ndarray): Anomaly map predicted via the model   (900, 900)
        threshold (float): Value to threshold anomaly scores into 0-1 range.
        kernel_size (int): Value to apply morphological operations to the predicted mask. Defaults to 4.

    Returns:
        Predicted anomaly mask
    """

    anomaly_map = anomaly_map.squeeze()
    mask: np.ndarray = np.zeros_like(anomaly_map).astype(np.uint8)
    mask[anomaly_map > threshold] = 1

    kernel = morphology.disk(kernel_size)
    mask = morphology.opening(mask, kernel)

    mask *= 255

    return mask


# from anomalib.deploy.inferencers.base_inferencer import Inferencer._superimpose_segmentation_mask
def gen_mask_border(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    """找mask的边缘,并显示在原图上

    Args:
        mask (np.ndarray):  mask
        image (np.ndarray): 原图

    Returns:
        np.ndarray: 原图上画上边缘
    """
    boundaries   = find_boundaries(mask)    # find_boundaries和dilation 返回和原图一样的 0 1 的图像(True False也可以)
    outlines     = dilation(boundaries, np.ones((3, 3)))
    mask_outline = image.copy()             # 深拷贝
    mask_outline[outlines] = [255, 0, 0]
    return mask_outline


def predict(yaml_path, image_path, model_dir):
    cfg = yaml.load(open(yaml_path,'r'), Loader=yaml.FullLoader)
    model = load_model(cfg, model_dir)
    image = load_image(image_path)
    image_size = [image.shape[0], image.shape[1]]
    transform = get_transform(cfg['DATASET']['resize'][0], cfg['DATASET']['resize'][1])
    x = transform(image=image)
    x = x['image'].unsqueeze(0)
    model.eval()
    with torch.inference_mode():
        y = model(x)        # [1, 2, H, W]
    y = F.softmax(y, dim=1) # [1, 2, H, W]
    y = y[0][1]             # [H, W]
    y = y.squeeze()         # [H, W]
    y = y.detach().cpu().numpy()

    # 还原到原图大小
    y = cv2.resize(y, (image_size[1], image_size[0]))

    # 1.计算mask
    mask = compute_mask(y)

    # 2.计算mask外边界
    mask_outline = gen_mask_border(mask, image)

    # 3.热力图混合原图
    superimposed_map = superimpose_anomaly_map(y, image)

    figsize = (4 * 9, 9)
    figure, axes = plt.subplots(1, 4, figsize=figsize)

    axes[0].imshow(image)
    axes[0].set_title("origin")
    axes[1].imshow(mask)
    axes[1].set_title("mask")
    axes[2].imshow(mask_outline)
    axes[2].set_title("mask_outline")
    axes[3].imshow(superimposed_map)
    axes[3].set_title("superimposed_map")
    plt.close()


if __name__=='__main__':
    # yaml_path  = "./configs/bottle.yaml"
    # image_path = "./datasets/MVTec/bottle/test/broken_large/000.png"
    # model_dir  = "./saved_model/MemSeg-bottle"
    # predict(yaml_path, image_path, model_dir)

    yaml_path  = "./configs/custom.yaml"
    image_path = "./datasets/custom/test/bad/002.jpg"
    model_dir  = "./saved_model/MemSeg-custom-256"
    predict(yaml_path, image_path, model_dir)
