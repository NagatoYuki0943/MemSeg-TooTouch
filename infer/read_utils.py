import numpy as np
import yaml
import json
import cv2
from typing import Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from skimage import morphology
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation


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
#   获取meta_data
#-----------------------------#
def get_json(path: str) -> dict:
    """get yaml config

    Args:
        path (str): json file path

    Returns:
        meta_data(dict): data
    """
    with open(path, mode='r', encoding='utf-8') as f:
        data = json.load(f)
    return data


#-----------------------------#
#   获取yaml config
#-----------------------------#
def get_yaml(path: str) -> dict:
    """get yaml config

    Args:
        path (str): yaml file path

    Returns:
        meta_data(dict): data
    """
    with open(path, mode='r', encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


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


def gen_images(image: np.ndarray, anomaly_map: np.ndarray, mask_thre: float = 0.5) -> list[np.ndarray]:
    """根据热力图,mask,热力图阈值生成mask,mask_outline,superimposed_map

    Args:
        save_path (str):            保存图片路径
        image (np.ndarray):         原图    ex: (900, 900, 3)
        anomaly_map (np.ndarray):   热力图  ex: (900, 900)
        mask_thre (float):          mask阈值. Defaults to 0.5.

    Returns:
        mask ,mask_outline, superimposed_map
    """
    # 1.计算mask
    mask = compute_mask(anomaly_map, mask_thre)

    # 2.计算mask外边界
    mask_outline = gen_mask_border(mask, image)

    # 3.热力图混合原图
    superimposed_map = superimpose_anomaly_map(anomaly_map, image)

    return [mask, mask_outline, superimposed_map]


def save_image(save_path: str, image: np.ndarray, mask: np.ndarray, mask_outline: np.ndarray, superimposed_map: np.ndarray, pred_score: float = 0.0):
    """保存图片

    Args:
        save_path (str):    保存路径
        image (np.ndarray): 原图
        mask (np.ndarray):  mask
        mask_outline (np.ndarray): mask边缘
        superimposed_map (np.ndarray): 热力图+原图
        pred_score (float): 预测得分. Defaults to 0.0
    """
    figsize = (4 * 9, 9)
    figure, axes = plt.subplots(1, 4, figsize=figsize)

    axes[0].imshow(image)
    axes[0].set_title("origin")
    axes[1].imshow(mask)
    axes[1].set_title("mask")
    axes[2].imshow(mask_outline)
    axes[2].set_title("outlines")
    axes[3].imshow(superimposed_map)
    axes[3].set_title("score: {:.4f}".format(pred_score))
    # plt.show()
    # return
    plt.savefig(save_path)
    plt.close()


def softmax(x: np.ndarray, axis=0) -> np.ndarray:
    """numpy softmax
    将每个值求e的指数全都变为大于0的值,然后除以求指数之后的总和
    :math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}`

    Args:
        x (np.ndarray): 计算的数据
        axis (int, optional): 在那个维度上计算. Defaults to 0.

    Returns:
        np.ndarray: 计算结果
    """
    # 为了稳定地计算softmax概率， 一般会减掉最大的那个元素
    x -= np.max(x, axis=axis, keepdims=True)
    # print(x)              # [-4. -3. -2. -1.  0.]
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


if __name__ == "__main__":
    pass
